"""Combined program for photonic crystal band gap optimization."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _enforce_symmetry(base):
    """
    Enforce square symmetry (D4) on a 32x32 array.
    Uses majority vote over 8 equivalent positions.
    """
    size = base.shape[0]
    b = np.array(base, dtype=np.int8)
    
    # Generate all 8 symmetries
    r0 = b
    r90 = np.rot90(b, 1)
    r180 = np.rot90(b, 2)
    r270 = np.rot90(b, 3)
    rx = b[:, ::-1]
    ry = b[::-1, :]
    rd = b.T
    rad = b[::-1, :].T # Anti-diagonal reflection
    
    votes = (r0 + r90 + r180 + r270 + rx + ry + rd + rad)
    cell = (votes >= 4).astype(np.int8)
    return cell

def _compute_gap_fast(cell, size=32, n_pw=10, eps_high=12.0, eps_low=1.0):
    """
    Compute the gap ratio between band 4 and 5 for TM polarization.
    Uses a fast plane-wave expansion. Reduced n_pw for speed.
    """
    cell_arr = np.array(cell, dtype=float)
    eps_arr = np.where(cell_arr > 0.5, eps_high, eps_low)
    inv_eps_arr = 1.0 / eps_arr
    
    # G vectors range from -n_pw to n_pw
    G_range = range(-n_pw, n_pw + 1)
    Gs = np.array([(mx, my) for mx in G_range for my in G_range], dtype=float)
    Gs *= 2.0 * np.pi
    nG = len(Gs)
    
    # Compute inv_eps_G for all G in Gs
    diff_range = range(-2*n_pw, 2*n_pw + 1)
    Gs_diff = np.array([(mx, my) for mx in diff_range for my in diff_range], dtype=float)
    Gs_diff *= 2.0 * np.pi
    nG_diff = len(Gs_diff)
    
    inv_eps_G_diff = np.zeros(nG_diff, dtype=complex)
    idxs = np.arange(size)
    
    # Vectorized FFT for faster coefficient calculation
    # We need coefficients for G = (mx, my) * 2pi
    # The array is size x size.
    # We can use FFT2D. 
    # F[kx, ky] = sum_{x,y} f[x,y] exp(-i 2pi (kx x + ky y) / size)
    # Our Gs_diff are multiples of 2pi.
    # So we just need the FFT of inv_eps_arr.
    
    # Zero pad to size_diff x size_diff to get all difference frequencies?
    # Actually, the difference frequencies G_i - G_j are in the range [-2n_pw, 2n_pw].
    # The FFT of the size x size array gives frequencies in [-size/2, size/2].
    # Since size=32 and 2*n_pw is usually < 16 (for n_pw=8), or even for n_pw=10 (20 < 16? No, 20 > 16).
    # Wait, if n_pw=10, max diff index is 20. Size is 32. Nyquist is 16.
    # So we must zero-pad to at least 41x41 to avoid aliasing?
    # Or just compute directly if n_pw is small.
    # For n_pw=10, 41x41 = 1681 coeffs. Direct loop is 1681 * 32*32 ops = ~1.7M ops. Very fast.
    
    for idx, (gx, gy) in enumerate(Gs_diff):
        mx_int = int(round(gx/(2*np.pi)))
        my_int = int(round(gy/(2*np.pi)))
        
        phase_x = np.exp(-1j * 2.0 * np.pi * mx_int * idxs / size)
        phase_y = np.exp(-1j * 2.0 * np.pi * my_int * idxs / size)
        phase_2d = phase_x[:, None] * phase_y[None, :]
        
        coeff = np.sum(inv_eps_arr * phase_2d) / (size * size)
        inv_eps_G_diff[idx] = coeff
        
    size_diff = 4 * n_pw + 1
    inv_eps_2d = np.zeros((size_diff, size_diff), dtype=complex)
    for idx, (gx, gy) in enumerate(Gs_diff):
        mx_int = int(round(gx/(2*np.pi)))
        my_int = int(round(gy/(2*np.pi)))
        inv_eps_2d[mx_int + 2*n_pw, my_int + 2*n_pw] = inv_eps_G_diff[idx]

    # 2. Define k-points in the irreducible Brillouin Zone
    k_points = []
    for t in np.linspace(0, 1, 6):
        k_points.append((t * np.pi, 0.0))
    for t in np.linspace(0, 1, 6):
        k_points.append((np.pi, t * np.pi))
    for t in np.linspace(0, 1, 6):
        k_points.append((t * np.pi, t * np.pi))
    # Fewer interior points for speed
    for t in np.linspace(0.2, 0.8, 3):
        for u in np.linspace(0.2, 0.8, 3):
             if u <= t:
                 k_points.append((t * np.pi, u * np.pi))

    max_band4 = 0.0
    min_band5 = np.inf
    
    dmx = (Gs[:, 0][:, None] - Gs[:, 0][None, :]) / (2*np.pi)
    dmy = (Gs[:, 1][:, None] - Gs[:, 1][None, :]) / (2*np.pi)
    dmx_int = np.round(dmx).astype(int)
    dmy_int = np.round(dmy).astype(int)
    imx = dmx_int + 2*n_pw
    imy = dmy_int + 2*n_pw
    inv_eps_mat = inv_eps_2d[imx, imy]

    for kx, ky in k_points:
        k = np.array([kx, ky])
        kGs = k[None, :] + Gs
        dot = np.einsum('ik,jk->ij', kGs, kGs)
        
        A = dot * inv_eps_mat
        A = (A + A.conj().T) / 2.0
        
        eigvals = np.linalg.eigvalsh(A)
        eigvals = np.real(eigvals)
        eigvals = eigvals[eigvals > 1e-10]
        
        if len(eigvals) >= 5:
            b4 = eigvals[3]
            b5 = eigvals[4]
            
            if b4 > max_band4:
                max_band4 = b4
            if b5 < min_band5:
                min_band5 = b5
                
    if min_band5 == np.inf or max_band4 <= 0:
        return -1.0
        
    gap = min_band5 - max_band4
    midgap = 0.5 * (min_band5 + max_band4)
    
    if midgap <= 0:
        return -1.0
        
    return 2.0 * gap / midgap

def _make_circle(radius, size=32, center_offset=0.0):
    base = np.zeros((size, size), dtype=np.int8)
    center = size / 2.0 + center_offset
    for i in range(size):
        for j in range(size):
            x = i - center + 0.5
            y = j - center + 0.5
            if x*x + y*y < radius*radius:
                base[i, j] = 1
    return _enforce_symmetry(base)

def _make_four_quadrant_circles(radius, offset, size=32):
    """
    Creates four small circles in the four quadrants.
    offset determines how far from the center they are.
    """
    base = np.zeros((size, size), dtype=np.int8)
    center = size / 2.0
    
    # Centers of the four circles
    centers = [
        (center - offset, center - offset),
        (center + offset, center - offset),
        (center - offset, center + offset),
        (center + offset, center + offset)
    ]
    
    for cx, cy in centers:
        for i in range(size):
            for j in range(size):
                x = i - cx + 0.5
                y = j - cy + 0.5
                if x*x + y*y < radius*radius:
                    base[i, j] = 1
                    
    return _enforce_symmetry(base)

def _make_cross(arm_width, length, size=32):
    base = np.zeros((size, size), dtype=np.int8)
    center = size / 2.0
    for i in range(size):
        for j in range(size):
            dx = abs(i - center + 0.5)
            dy = abs(j - center + 0.5)
            if dx < length and dy < arm_width / 2.0:
                base[i, j] = 1
            elif dy < length and dx < arm_width / 2.0:
                base[i, j] = 1
    return _enforce_symmetry(base)

def design(problem):
    """
    Design the photonic crystal unit cell.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12.0)
    eps_low = problem.get("eps_low", 1.0)
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    candidates = []
    
    # 1. Standard single circle (baseline)
    for r in [0.15, 0.2, 0.25, 0.3]:
        candidates.append(_make_circle(r * size, size))
        
    # 2. Four quadrant circles
    for offset in [0.1, 0.15, 0.2, 0.25]:
        for r in [0.05, 0.08, 0.1, 0.12]:
            if offset + r < size/2:
                candidates.append(_make_four_quadrant_circles(r * size, offset * size, size))
                
    # 3. Crosses
    for aw in [2, 4, 6, 8]:
        for ln in [4, 8, 12]:
            candidates.append(_make_cross(aw, ln, size))
            
    # 4. Inverted structures
    def _make_inverted_circle(radius, size=32):
        base = np.ones((size, size), dtype=np.int8)
        center = size / 2.0
        for i in range(size):
            for j in range(size):
                x = i - center + 0.5
                y = j - center + 0.5
                if x*x + y*y < radius*radius:
                    base[i, j] = 0
        return _enforce_symmetry(base)
        
    def _make_inverted_four_quadrant_circles(radius, offset, size=32):
        base = np.ones((size, size), dtype=np.int8)
        center = size / 2.0
        centers = [
            (center - offset, center - offset),
            (center + offset, center - offset),
            (center - offset, center + offset),
            (center + offset, center + offset)
        ]
        for cx, cy in centers:
            for i in range(size):
                for j in range(size):
                    x = i - cx + 0.5
                    y = j - cy + 0.5
                    if x*x + y*y < radius*radius:
                        base[i, j] = 0
        return _enforce_symmetry(base)

    for r in [0.15, 0.2, 0.25]:
        candidates.append(_make_inverted_circle(r * size, size))
        
    for offset in [0.15, 0.2]:
        for r in [0.08, 0.1]:
             if offset + r < size/2:
                 candidates.append(_make_inverted_four_quadrant_circles(r * size, offset * size, size))

    # Evaluate Candidates
    best_cell = None
    best_score = -1.0
    
    for cell in candidates:
        if time.time() > deadline:
            break
        try:
            score = _compute_gap_fast(cell, size, n_pw=10, eps_high=eps_high, eps_low=eps_low)
        except Exception:
            score = -1.0
            
        if score > best_score:
            best_score = score
            best_cell = cell
            
    if best_cell is None:
        best_cell = _make_circle(0.2 * size, size)
        
    # 2. Parametric Local Search
    # We will refine the best cell by treating it as a parametric geometry.
    # Since we don't know the exact parameters of the best cell, we will try to
    # optimize a generic "ring + cross" or "quadrant circles" model around the best cell.
    # However, a simpler and more robust approach for local search on binary pixels
    # that respects symmetry is to try flipping specific symmetric groups of pixels.
    
    # Let's use a smarter local search: 
    # 1. Identify the best cell.
    # 2. Try small perturbations that maintain symmetry.
    # Instead of random flips, we will try "morphing" the best cell towards other promising candidates
    # or simply do a few rounds of random symmetric flips with a lower n_pw.
    
    current_cell = np.array(best_cell, dtype=np.int8)
    current_score = best_score
    rng = np.random.default_rng(42)
    
    # Use n_pw=10 for local search to be fast
    search_n_pw = 10
    
    # Local search: try flipping 1-3 pixels, enforce symmetry, evaluate.
    # Limit iterations to ensure we don't run out of time.
    max_iterations = 500
    iter_count = 0
    
    while time.time() < deadline and iter_count < max_iterations:
        iter_count += 1
        mut_cell = current_cell.copy()
        
        # Flip 1-2 random pixels
        n_flips = rng.integers(1, 3)
        for _ in range(n_flips):
            i = rng.integers(0, size)
            j = rng.integers(0, size)
            mut_cell[i, j] = 1 - mut_cell[i, j]
            
        # Enforce symmetry
        sym_cell = _enforce_symmetry(mut_cell)
        
        # Evaluate
        try:
            score = _compute_gap_fast(sym_cell, size, n_pw=search_n_pw, eps_high=eps_high, eps_low=eps_low)
        except Exception:
            score = -1.0
            
        if score > current_score:
            current_cell = np.array(sym_cell, dtype=np.int8)
            current_score = score
            # Reset iteration count or continue? Continue to allow further improvements.
            
    return current_cell.tolist()

# EVOLVE-BLOCK-END
