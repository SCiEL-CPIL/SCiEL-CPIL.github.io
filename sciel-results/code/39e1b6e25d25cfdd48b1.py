"""Design a 32x32 symmetric photonic crystal unit cell for maximum TM band gap."""
import time
import math
import numpy as np

# Deadline guard: evaluator kills at 230s wall-clock; keep margin
_DEADLINE = 220.0  # seconds
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _DEADLINE - _elapsed()


def _enforce_d4(cell):
    """Enforce full D4 symmetry on the cell."""
    size = len(cell)
    result = [[0] * size for _ in range(size)]
    
    for i in range(size):
        for j in range(size):
            # Get all 8 symmetric positions
            positions = [
                (i, j),
                (size - 1 - i, j),
                (i, size - 1 - j),
                (size - 1 - i, size - 1 - j),
                (j, i),
                (size - 1 - j, i),
                (j, size - 1 - i),
                (size - 1 - j, size - 1 - i),
            ]
            # Majority vote
            vals = [cell[pi][pj] for pi, pj in positions]
            result[i][j] = 1 if sum(vals) >= 4 else 0
    
    return result


def _compute_tm_gap_fast(cell, N_pw=13, n_kpts=10):
    """
    Compute the TM band gap between bands 4 and 5.
    Uses plane-wave expansion with exact Fourier coefficients.
    Optimized for speed during optimization loops.
    """
    size = len(cell)
    eps_high = 12.0
    eps_low = 1.0
    
    # Precompute 1/eps for each pixel
    inv_eps = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            inv_eps[i][j] = 1.0 / (eps_high if cell[i][j] == 1 else eps_low)
    
    # Compute Fourier coefficients for m, n in range [-N+1, N-1]
    # Using vectorized operations for speed
    m_vals = np.arange(-N_pw + 1, N_pw)
    n_vals = np.arange(-N_pw + 1, N_pw)
    
    # x, y coordinates of pixel centers
    x_coords = (np.arange(size) + 0.5) / size
    y_coords = (np.arange(size) + 0.5) / size
    
    # Create meshgrid
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    coeffs = {}
    for m in m_vals:
        for n in n_vals:
            # phase = exp(-i * 2 * pi * (m * x + n * y))
            phase = np.exp(-1j * 2 * np.pi * (m * X + n * Y))
            c = np.sum(inv_eps * phase) / (size * size)
            coeffs[(m, n)] = c
    
    # Generate k-points in the irreducible BZ
    # Irreducible BZ for square lattice: 0 <= ky <= kx, kx + ky <= pi
    kpts = []
    # Sample points along high-symmetry lines and interior
    # (0,0), (pi,0), (pi,pi) are the vertices of the full BZ, but irreducible is 1/8
    # Vertices of irreducible triangle: (0,0), (pi,0), (pi/2, pi/2) is not quite right.
    # Standard irreducible wedge for square: 0 <= ky <= kx, kx <= pi, ky <= pi, kx+ky <= pi?
    # Actually, the first BZ is [-pi, pi]x[-pi, pi].
    # The irreducible sector is 0 <= ky <= kx and kx + ky <= pi.
    
    # Let's sample a grid in this triangle
    n_samples = 4
    for i in range(n_samples + 1):
        for j in range(n_samples + 1):
            kx = (i / n_samples) * np.pi
            ky = (j / n_samples) * np.pi
            if ky <= kx and kx + ky <= np.pi:
                kpts.append((kx, ky))
    
    # Ensure we have some specific high-symmetry points
    kpts.append((0, 0))
    kpts.append((np.pi/2, 0))
    kpts.append((np.pi/2, np.pi/2))
    kpts.append((np.pi/4, np.pi/4))
    
    # Filter unique
    unique_kpts = []
    seen = set()
    for kx, ky in kpts:
        key = (round(kx, 4), round(ky, 4))
        if key not in seen:
            seen.add(key)
            unique_kpts.append((kx, ky))
    
    kpts = unique_kpts[:n_kpts]
    
    max_omega4 = 0.0
    min_omega5 = float('inf')
    
    g_vectors = []
    for m in m_vals:
        for n in n_vals:
            g_vectors.append((2 * np.pi * m, 2 * np.pi * n, m, n))
    
    npw = len(g_vectors)
    
    for kx, ky in kpts:
        A = np.zeros((npw, npw), dtype=complex)
        
        for mi in range(npw):
            gx_m, gy_m, mm, mn = g_vectors[mi]
            kx_m = kx + gx_m
            ky_m = ky + gy_m
            km2 = kx_m * kx_m + ky_m * ky_m
            
            for ni in range(npw):
                gx_n, gy_n, nn, nnn = g_vectors[ni]
                
                dm = mm - nn
                dn = mn - nnn
                
                c = coeffs.get((dm, dn), 0.0)
                
                A[mi, ni] = c * km2
        
        try:
            eigenvalues = np.linalg.eigvalsh(A)
            eigenvalues = np.real(eigenvalues)
            eigenvalues = np.sort(eigenvalues)
            
            if len(eigenvalues) >= 5:
                omega4_sq = eigenvalues[3]
                omega5_sq = eigenvalues[4]
                
                if omega4_sq > 0 and omega5_sq > 0:
                    omega4 = np.sqrt(omega4_sq)
                    omega5 = np.sqrt(omega5_sq)
                    
                    max_omega4 = max(max_omega4, omega4)
                    min_omega5 = min(min_omega5, omega5)
        except:
            continue
    
    if min_omega5 == float('inf') or max_omega4 == 0:
        return 0.0
    
    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0
    
    if midgap <= 0:
        return 0.0
    
    return 2.0 * gap / midgap


def _hill_climb(cell, max_iters=100):
    """
    Perform local search by flipping pixels.
    """
    size = len(cell)
    best_cell = [row[:] for row in cell]
    best_score = _compute_tm_gap_fast(best_cell)
    
    # Identify independent pixels (fundamental domain)
    # D4 symmetry: we only need to define pixels where i <= j and i+j < size
    # Actually, the orbit size is 8 for generic points.
    # Let's just pick random pixels, flip them, enforce symmetry, and check score.
    
    rng = np.random.default_rng(42)
    
    for _ in range(max_iters):
        if _remaining() < 10:
            break
            
        # Pick a random pixel to flip
        i = rng.integers(0, size)
        j = rng.integers(0, size)
        
        new_cell = [row[:] for row in best_cell]
        # Flip the pixel and all its symmetric counterparts
        positions = [
            (i, j),
            (size - 1 - i, j),
            (i, size - 1 - j),
            (size - 1 - i, size - 1 - j),
            (j, i),
            (size - 1 - j, i),
            (j, size - 1 - i),
            (size - 1 - j, size - 1 - i),
        ]
        
        current_val = new_cell[i][j]
        for pi, pj in positions:
            new_cell[pi][pj] = 1 - current_val
            
        new_score = _compute_tm_gap_fast(new_cell)
        
        if new_score > best_score:
            best_score = new_score
            best_cell = new_cell
    
    return best_cell, best_score


def design(problem):
    """
    Design a 32x32 symmetric photonic crystal unit cell.
    Returns a nested list of 0/1 pixels.
    """
    global _start_time
    _start_time = time.time()
    
    size = problem.get("pixels", 32)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Central disk
    cell1 = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x * x + y * y < 0.25 * 0.25:
                cell1[i][j] = 1
    candidates.append(_enforce_d4(cell1))
    
    # 2. Ring
    cell2 = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = math.sqrt(x * x + y * y)
            if 0.2 < r < 0.3:
                cell2[i][j] = 1
    candidates.append(_enforce_d4(cell2))
    
    # 3. Grid of holes (checkerboard-like)
    cell3 = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            # Small holes at grid points
            if abs(x) < 0.1 and abs(y) < 0.1:
                cell3[i][j] = 1
            if abs(x) < 0.1 and abs(y - 0.25) < 0.1:
                cell3[i][j] = 1
            if abs(x) < 0.1 and abs(y + 0.25) < 0.1:
                cell3[i][j] = 1
            if abs(x - 0.25) < 0.1 and abs(y) < 0.1:
                cell3[i][j] = 1
            if abs(x + 0.25) < 0.1 and abs(y) < 0.1:
                cell3[i][j] = 1
    candidates.append(_enforce_d4(cell3))
    
    # 4. Random noise
    cell4 = [[int(np.random.rand() > 0.5) for _ in range(size)] for _ in range(size)]
    candidates.append(_enforce_d4(cell4))
    
    best_cell = None
    best_score = -float('inf')
    
    for cand in candidates:
        if _remaining() < 20:
            break
        
        # Run hill climbing
        refined_cell, score = _hill_climb(cand, max_iters=50)
        
        if score > best_score:
            best_score = score
            best_cell = refined_cell
    
    if best_cell is None:
        # Fallback
        best_cell = candidates[0]
    
    return best_cell
