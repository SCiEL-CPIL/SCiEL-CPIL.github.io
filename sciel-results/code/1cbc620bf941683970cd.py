import numpy as np
import time
import math

_DEADLINE = 220.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

# ---- Fast plane-wave TM surrogate (5x5 = 25 modes) ----
def _build_pw_evaluator(size=32, K=2):
    """Precompute everything needed for fast TM band evaluation."""
    G = [(h, k) for h in range(-K, K + 1) for k in range(-K, K + 1)]
    N = len(G)
    Gx = np.array([g[0] for g in G], dtype=np.float64)
    Gy = np.array([g[1] for g in G], dtype=np.float64)
    
    # Precompute G differences for FFT indexing
    Gdiff_h = np.zeros((N, N), dtype=int)
    Gdiff_k = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            Gdiff_h[i, j] = (G[i][0] - G[j][0]) % size
            Gdiff_k[i, j] = (G[i][1] - G[j][1]) % size
    
    # Reduced K-points in irreducible BZ for speed (Gamma, M, X, and one interior)
    k_points = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.25, 0.25),
        (0.0, 0.5)
    ]
    
    kx_arr = np.array([p[0] for p in k_points])
    ky_arr = np.array([p[1] for p in k_points])
    nk = len(k_points)
    
    def evaluate(cell_grid):
        """Compute gap-midgap ratio for bands 4 and 5."""
        eps_grid = np.where(cell_grid == 1, 12.0, 1.0)
        inv_eps = 1.0 / eps_grid
        fft_1eps = np.fft.fft2(inv_eps)
        # coeffs[i,j] corresponds to G_i - G_j
        coeffs = fft_1eps[Gdiff_h, Gdiff_k] / (size * size)
        coeffs_real = coeffs.real
        
        max_w4 = 0.0
        min_w5 = float('inf')
        
        for idx in range(nk):
            kx = kx_arr[idx]
            ky = ky_arr[idx]
            Kx = Gx + kx
            Ky = Gy + ky
            # H_ij = (K_i . K_j) * c_{i-j}
            dots = np.outer(Kx, Kx) + np.outer(Ky, Ky)
            H = dots * coeffs_real
            H = 0.5 * (H + H.T)
            eigenvalues = np.linalg.eigvalsh(H)
            if len(eigenvalues) >= 5:
                w4 = np.sqrt(max(0.0, eigenvalues[3]))
                w5 = np.sqrt(max(0.0, eigenvalues[4]))
                if w4 > max_w4:
                    max_w4 = w4
                if w5 < min_w5:
                    min_w5 = w5
        
        if min_w5 == float('inf') or max_w4 <= 0 or min_w5 <= 0:
            return -1.0
        return 2.0 * (min_w5 - max_w4) / (min_w5 + max_w4)
    
    return evaluate

# ---- C4v symmetry enforcement ----
def _enforce_c4v(cell, size=32):
    """Enforce C4v symmetry: cell == cell.T, cell == cell[::-1,:], cell == cell[:,::-1]."""
    c = np.asarray(cell, dtype=np.float64)
    sym = c.copy()
    sym = sym + sym.T
    sym = sym + sym[::-1, :]
    sym = sym + sym[:, ::-1]
    sym = sym + sym.T[::-1, :]
    sym = sym + sym.T[:, ::-1]
    sym = sym + sym[::-1, :].T
    sym = sym + sym[:, ::-1].T
    return (sym >= 4).astype(int)

# ---- Generate a good initial guess ----
def _initial_guess(size=32):
    """Create a structure with 4 holes in the center, optimized for TM gap."""
    cell = np.zeros((size, size), dtype=int)
    # Try a configuration of 4 circular holes
    # Center of cell is (15.5, 15.5)
    # Place holes at (8,8), (23,8), (8,23), (23,23) approx
    centers = [(8, 8), (23, 8), (8, 23), (23, 23)]
    radius = 3.0
    yy, xx = np.mgrid[0:size, 0:size]
    for cx, cy in centers:
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        cell[dist < radius] = 0 # Holes are air (0), background is dielectric (1)
    
    # Set background to dielectric (1)
    cell = 1 - cell # Invert: 1 is dielectric, 0 is air
    # Ensure the holes are actually 0
    for cx, cy in centers:
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        cell[dist < radius] = 0
        
    return _enforce_c4v(cell, size)

# ---- Main design function ----
def design(problem):
    size = problem["pixels"]  # 32
    
    start_time = time.time()
    deadline = start_time + 210.0  # Safety margin
    
    # Build fast evaluator
    evaluate = _build_pw_evaluator(size, K=2)
    
    # Start with a good initial guess
    best_cell = _initial_guess(size)
    best_score = evaluate(best_cell)
    
    # Simulated Annealing on pixel orbits
    # Precompute orbits for fast flipping
    orbits = []
    visited = set()
    for i in range(size):
        for j in range(size):
            if (i, j) in visited:
                continue
            orbit = [
                (i, j),
                (size - 1 - i, j),
                (i, size - 1 - j),
                (size - 1 - i, size - 1 - j),
                (j, i),
                (size - 1 - j, j),
                (j, size - 1 - i),
                (size - 1 - j, size - 1 - i),
            ]
            unique_pts = []
            for p in orbit:
                if p not in unique_pts:
                    unique_pts.append(p)
            orbits.append(unique_pts)
            for p in unique_pts:
                visited.add(p)
    
    n_orbits = len(orbits)
    current_cell = best_cell.copy()
    current_score = best_score
    
    temperature = 0.1
    cooling_rate = 0.999
    min_temp = 0.01
    
    # Main optimization loop
    while time.time() < deadline - 5.0:
        # Pick a random orbit
        oi = np.random.randint(n_orbits)
        orbit = orbits[oi]
        
        # Create new cell by flipping this orbit
        new_cell = current_cell.copy()
        for (i, j) in orbit:
            new_cell[i, j] = 1 - new_cell[i, j]
        
        new_score = evaluate(new_cell)
        
        # Metropolis criterion
        delta = new_score - current_score
        if delta > 0 or (temperature > min_temp and np.random.random() < math.exp(delta / temperature)):
            current_cell = new_cell
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_cell = current_cell.copy()
        
        # Cool down
        temperature = max(temperature * cooling_rate, min_temp)
        
        # Check time every 100 iterations
        if oi % 100 == 0 and time.time() > deadline - 10.0:
            break
    
    # Finalize: enforce symmetry and return
    final_cell = _enforce_c4v(best_cell, size)
    return final_cell.tolist()
