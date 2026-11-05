import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _enforce_symmetry(size, base):
    """
    Enforces full C4v symmetry (square symmetry group).
    Input: base (size x size array of 0/1)
    Output: cell (size x size list of lists of 0/1)
    """
    b = np.array(base, dtype=np.int8)
    N = size
    cell = np.zeros((N, N), dtype=np.int8)
    processed = np.zeros((N, N), dtype=bool)
    
    for i in range(N):
        for j in range(N):
            if processed[i, j]:
                continue
            
            # Generate the 8 symmetric positions for (i, j)
            orbit = [
                (i, j),
                (N-1-i, j),
                (j, i),
                (N-1-i, N-1-j),
                (i, N-1-j),
                (j, N-1-i),
                (N-1-j, i),
                (N-1-j, N-1-i)
            ]
            
            # Remove duplicates
            unique_orbit = list(set(orbit))
            
            # Majority vote from input base
            vals = [b[r, c] for r, c in unique_orbit]
            if sum(vals) >= len(vals) / 2.0:
                val = 1
            else:
                val = 0
            
            # Assign value to all unique positions in the orbit
            for r, c in unique_orbit:
                cell[r, c] = val
                processed[r, c] = True
                
    return cell.tolist()

def _compute_surrogate_gap(cell, size, n_pw=8, n_kx=5, n_ky=5, eps_high=12.0, eps_low=1.0):
    """
    Fast plane-wave TM band calculation surrogate.
    n_pw=8 gives 17x17=289 plane waves, matching evaluator.
    """
    # Build reciprocal lattice vectors
    G_range = range(-n_pw, n_pw + 1)
    Gs = np.array([(mx, my) for mx in G_range for my in G_range], dtype=float)
    nG = len(Gs)
    
    # Compute inv_eps Fourier coefficients using FFT
    cell_arr = np.array(cell, dtype=float)
    eps_arr = np.where(cell_arr > 0.5, eps_high, eps_low)
    inv_eps_arr = 1.0 / eps_arr
    
    inv_eps_fft = np.fft.fft2(inv_eps_arr)
    
    # Map G indices to FFT bin indices
    G_bins = np.array([(mx % size, my % size) for mx, my in Gs], dtype=int)
    
    # Build inv_eps matrix: M[i,j] = inv_eps(G_i - G_j)
    dmx = Gs[:, 0][:, None] - Gs[:, 0][None, :]
    dmy = Gs[:, 1][:, None] - Gs[:, 1][None, :]
    
    diff_bins_row = dmx % size
    diff_bins_col = dmy % size
    
    # inv_eps_mat[i, j] = inv_eps_fft[diff_bins_row[i, j], diff_bins_col[i, j]]
    inv_eps_mat = inv_eps_fft[diff_bins_row, diff_bins_col]
    
    # K-points: Irreducible BZ of square lattice
    # Triangle: 0 <= ky <= kx <= pi
    k_points = []
    k_points.append((0.0, 0.0))
    # Line Gamma-X (kx from 0 to pi, ky=0)
    for t in np.linspace(0.1, 1.0, n_kx):
        k_points.append((t * np.pi, 0.0))
    # Line Gamma-M (kx=ky, from 0 to pi)
    for t in np.linspace(0.1, 1.0, n_ky):
        k_points.append((t * np.pi, t * np.pi))
    # Line X-M (kx=pi, ky from 0 to pi)
    for t in np.linspace(0.1, 1.0, n_kx):
        k_points.append((np.pi, t * np.pi))
    
    max_band4 = 0.0
    min_band5 = np.inf
    
    for kx, ky in k_points:
        k = np.array([kx, ky])
        kGs = k + Gs
        
        # Compute dot products: (k+Gi)·(k+Gj)
        kGs_x = kGs[:, 0]
        kGs_y = kGs[:, 1]
        dot = np.outer(kGs_x, kGs_x) + np.outer(kGs_y, kGs_y)
        
        A = dot * inv_eps_mat
        # A should be Hermitian
        A = (A + A.conj().T) / 2.0
        
        # Solve eigenvalue problem
        eigvals = np.linalg.eigvalsh(A)
        eigvals = np.real(eigvals)
        # Keep only positive eigenvalues
        eigvals = eigvals[eigvals > 1e-10]
        
        if len(eigvals) >= 5:
            b4 = eigvals[3]
            b5 = eigvals[4]
            max_band4 = max(max_band4, b4)
            min_band5 = min(min_band5, b5)
    
    if min_band5 == np.inf or max_band4 == 0:
        return -1.0
    
    gap = min_band5 - max_band4
    midgap = 0.5 * (min_band5 + max_band4)
    if midgap <= 0:
        return -1.0
    return 2.0 * gap / midgap

def _generate_parametric_cells(size):
    """Generate a diverse set of parametric candidates."""
    candidates = []
    center = size // 2
    
    def make_cell_from_mask(mask_func, bg_val, shape_val):
        base = np.full((size, size), bg_val, dtype=np.int8)
        for i in range(size):
            for j in range(size):
                if mask_func(i, j):
                    base[i, j] = shape_val
        return _enforce_symmetry(size, base)
    
    # 1. Single central hole in dielectric (bg=1, hole=0)
    for r in np.arange(2.0, 8.0, 0.5):
        def mask(i, j, r=r):
            dx = i - center
            dy = j - center
            return dx*dx + dy*dy <= r*r
        candidates.append(make_cell_from_mask(mask, 1, 0))
        
    # 2. Four holes in dielectric
    for d in np.arange(4.0, 10.0, 0.5):
        for r in np.arange(1.0, 4.0, 0.5):
            def mask(i, j, d=d, r=r):
                dx = i - center
                dy = j - center
                # 4 holes at (d,0), (-d,0), (0,d), (0,-d)
                if dx*dx + (dy-d)*(dy-d) <= r*r: return True
                if dx*dx + (dy+d)*(dy+d) <= r*r: return True
                if (dx-d)*(dx-d) + dy*dy <= r*r: return True
                if (dx+d)*(dx+d) + dy*dy <= r*r: return True
                return False
            candidates.append(make_cell_from_mask(mask, 1, 0))
            
    # 3. Cross shape in dielectric
    for w in np.arange(1.0, 5.0, 0.5):
        for l in np.arange(5.0, 12.0, 0.5):
            def mask(i, j, w=w, l=l):
                dx = abs(i - center)
                dy = abs(j - center)
                if dx < l and dy < w: return True
                if dy < l and dx < w: return True
                return False
            candidates.append(make_cell_from_mask(mask, 0, 1))
            
    # 4. Rounded square hole
    for r_corner in np.arange(1.0, 4.0, 0.5):
        for r_side in np.arange(3.0, 7.0, 0.5):
            def mask(i, j, rc=r_corner, rs=r_side):
                dx = abs(i - center)
                dy = abs(j - center)
                if dx < rs and dy < rs:
                    # Check if in corner regions
                    if dx > rs - rc and dy > rs - rc:
                        cdx = dx - (rs - rc)
                        cdy = dy - (rs - rc)
                        return cdx*cdx + cdy*cdy <= rc*rc
                    return True
                return False
            candidates.append(make_cell_from_mask(mask, 1, 0))

    return candidates

def design(problem):
    """Time-bounded parametric search for TM band gap maximization."""
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12.0)
    eps_low = problem.get("eps_low", 1.0)
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Generate initial candidates
    candidates = _generate_parametric_cells(size)
    
    best_cell = None
    best_score = -1.0
    
    # Evaluate initial candidates
    for cell in candidates:
        if time.time() > deadline:
            break
        try:
            score = _compute_surrogate_gap(cell, size, n_pw=8, n_kx=5, n_ky=5, eps_high=eps_high, eps_low=eps_low)
        except Exception:
            score = -1.0
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # Local search: refine the best candidate with small perturbations
    if best_cell is not None:
        current_cell = best_cell
        current_score = best_score
        rng = np.random.default_rng(42)
        
        # Perform a few rounds of local search
        for round_num in range(5):
            if time.time() > deadline:
                break
            improved = False
            for step in range(20):
                if time.time() > deadline:
                    break
                # Random mutation: flip some pixels
                mut_cell = np.array(current_cell, dtype=np.int8)
                n_flips = rng.integers(1, 4)
                for _ in range(n_flips):
                    i = rng.integers(0, size)
                    j = rng.integers(0, size)
                    mut_cell[i, j] = 1 - mut_cell[i, j]
                sym_cell = _enforce_symmetry(size, mut_cell)
                
                try:
                    score = _compute_surrogate_gap(sym_cell, size, n_pw=8, n_kx=5, n_ky=5, eps_high=eps_high, eps_low=eps_low)
                except Exception:
                    score = -1.0
                
                if score > current_score:
                    current_cell = sym_cell
                    current_score = score
                    improved = True
                    if score > best_score:
                        best_score = score
                        best_cell = sym_cell
            
            if not improved:
                break
    
    if best_cell is None:
        # Fallback to a simple circle
        base = np.zeros((size, size), dtype=np.int8)
        center = size // 2
        for i in range(size):
            for j in range(size):
                dx = i - center
                dy = j - center
                if dx*dx + dy*dy < 16:
                    base[i, j] = 1
        best_cell = _enforce_symmetry(size, base)
    
    return best_cell

# EVOLVE-BLOCK-END
