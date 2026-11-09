import numpy as np
import time
import math
from scipy.optimize import minimize

def design(problem):
    """
    Designs a 32x32 unit cell for a TM band gap between bands 4 and 5.
    Strategy:
    1. Global topology search using parameterized geometric primitives (disks).
    2. High-fidelity 17x17 plane-wave evaluator.
    3. Local pixel-level refinement (Hill Climbing) on the best global candidate.
    """
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 190.0

    # --- 1. Setup Plane Wave Evaluator (17x17) ---
    pw_half = 8 # 17x17 = 289 modes
    G_vectors = [(h, k) for h in range(-pw_half, pw_half + 1) 
                 for k in range(-pw_half, pw_half + 1)]
    N_pw = len(G_vectors)
    
    Gx_arr = np.array([g[0] for g in G_vectors], dtype=float)
    Gy_arr = np.array([g[1] for g in G_vectors], dtype=float)
    
    # Precompute G_diffs indices for FFT
    # G_diff = G_i - G_j
    # We need coeffs of 1/eps at G_i - G_j
    # FFT of 1/eps gives coeffs at integer frequencies
    # Index in FFT array: (h % size, k % size)
    
    # We will compute FFT once per cell evaluation
    
    # K-points in irreducible BZ
    # Triangle: (0,0) -> (0.5, 0) -> (0.5, 0.5) -> (0,0)
    k_points = []
    n_k = 30
    # Along Gamma-M: (0, y)
    for i in range(n_k // 3 + 1):
        t = 0.5 * i / (n_k // 3)
        k_points.append((0.0, t))
    # Along M-X: (x, 0.5-x) ? No, M is (0.5, 0). X is (0.5, 0.5).
    # M->X: (0.5, y) where y goes 0 to 0.5?
    # Wait, irreducible BZ of square lattice is the triangle (0,0)-(0.5,0)-(0.5,0.5).
    # Boundaries:
    # 1. Gamma-M: (0, y), y in [0, 0.5]
    # 2. M-X: (0.5, y), y in [0, 0.5] ?? No.
    # The BZ is the square [-0.5, 0.5] x [-0.5, 0.5].
    # Irreducible part is 1/8 of it? No, 1/8 is for C4v.
    # The sector is (0,0) to (0.5, 0) to (0.5, 0.5).
    # Edges:
    # 1. (0,0) to (0.5, 0): kx in [0, 0.5], ky = 0.
    # 2. (0.5, 0) to (0.5, 0.5): kx = 0.5, ky in [0, 0.5].
    # 3. (0.5, 0.5) to (0,0): kx = ky, kx in [0, 0.5].
    
    k_points = []
    # 10 points on Gamma-M
    for t in np.linspace(0, 0.5, 10):
        k_points.append((t, 0.0))
    # 10 points on M-X
    for t in np.linspace(0, 0.5, 10):
        k_points.append((0.5, t))
    # 10 points on X-Gamma
    for t in np.linspace(0, 0.5, 10):
        k_points.append((0.5 - t, 0.5 - t)) # This is X to Gamma?
        # X is (0.5, 0.5). Gamma is (0,0).
        # Vector is (-0.5, -0.5).
        # Point = (0.5, 0.5) + t * (-1, -1) ? No.
        # Point = (0.5(1-t), 0.5(1-t)) for t in [0,1].
        # At t=0, (0.5, 0.5). At t=1, (0,0).
    # Correcting X-Gamma loop:
    k_points = []
    for t in np.linspace(0, 0.5, 10):
        k_points.append((t, 0.0))
    for t in np.linspace(0, 0.5, 10):
        k_points.append((0.5, t))
    for t in np.linspace(0, 0.5, 10):
        k_points.append((0.5 - t, 0.5 - t))
        
    # Remove duplicates
    k_points = list(set(k_points))
    
    def evaluate_cell(cell_grid):
        """
        Computes the gap-midgap ratio for bands 4 and 5.
        cell_grid: 32x32 array of 0/1
        """
        eps_grid = np.where(cell_grid == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_grid
        
        # FFT of 1/eps
        fft_1eps = np.fft.fft2(inv_eps)
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            # K vectors for each plane wave
            Kx = Gx_arr + kx
            Ky = Gy_arr + ky
            
            # H_ij = (K_i dot K_j) * coeff_{G_i - G_j}
            # We need coeff_{G_i - G_j}
            # G_i - G_j = (h_i - h_j, k_i - k_j)
            # FFT index: ( (h_i - h_j) % size, (k_i - k_j) % size )
            
            # Vectorized construction of H
            # This loop is the bottleneck.
            # N_pw = 289.
            # We can precompute the indices for G_diffs.
            
            H = np.zeros((N_pw, N_pw), dtype=float)
            
            # To speed up, we can precompute the G_diff indices once.
            # But G_diff depends on i, j.
            # Let's precompute a 2D array of indices for all i,j pairs.
            # This is 289x289x2 ints. ~ 200KB. Fine.
            
            # We'll do this outside the loop if possible, but G_vectors are fixed.
            
            for i in range(N_pw):
                hi, ki = G_vectors[i]
                Kx_i = Kx[i]
                Ky_i = Ky[i]
                
                # Dot products with all j
                # K_i dot K_j = Kx_i * Kx_j + Ky_i * Ky_j
                dots_i = Kx_i * Kx + Ky_i * Ky
                
                # Coeffs for G_i - G_j
                # h_diff = hi - h_j
                # k_diff = ki - k_j
                h_diffs = hi - Gx_arr.astype(int)
                k_diffs = ki - Gy_arr.astype(int)
                
                # Map to FFT indices
                idx_h = h_diffs % size
                idx_k = k_diffs % size
                
                coeffs = fft_1eps[idx_h, idx_k] / (size * size)
                
                H[i, :] = dots_i * coeffs.real
            
            # Hermitian matrix
            # H should be Hermitian.
            # eigvalsh requires Hermitian.
            # Due to floating point, we might need to symmetrize.
            H = (H + H.T) / 2.0
            
            eigenvalues = np.linalg.eigvalsh(H)
            
            if len(eigenvalues) >= 5:
                w4_sq = eigenvalues[3]
                w5_sq = eigenvalues[4]
                
                w4 = np.sqrt(max(0, w4_sq))
                w5 = np.sqrt(max(0, w5_sq))
                
                if w4 > max_omega_4:
                    max_omega_4 = w4
                if w5 < min_omega_5:
                    min_omega_5 = w5
        
        if min_omega_5 == float('inf'):
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        ratio = 2 * gap / midgap
        return ratio

    # --- 2. Parameterized Cell Construction ---
    
    def build_cell(params, invert=False):
        """
        params: [r_center, r_corner, d_corner, r_edge, d_edge]
        """
        r_center = params[0]
        r_corner = params[1]
        d_corner = params[2]
        r_edge = params[3]
        d_edge = params[4]
        
        cell = np.zeros((size, size), dtype=int)
        
        # Helper to add disk
        def add_disk(cx, cy, r):
            if r <= 0: return
            min_i = max(0, int(cy - r))
            max_i = min(size - 1, int(cy + r) + 1)
            min_j = max(0, int(cx - r))
            max_j = min(size - 1, int(cx + r) + 1)
            
            for i in range(min_i, max_i + 1):
                for j in range(min_j, max_j + 1):
                    dist = math.sqrt((i - cy)**2 + (j - cx)**2)
                    if dist <= r:
                        cell[i, j] = 1
                        
        # Center disk
        add_disk(16, 16, r_center)
        
        # Corner disks
        # Centers at (d, d), (d, 32-d), etc.
        # d_corner is distance from corner (0,0)
        c1 = d_corner
        c2 = size - 1 - d_corner
        
        add_disk(c1, c1, r_corner)
        add_disk(c1, c2, r_corner)
        add_disk(c2, c1, r_corner)
        add_disk(c2, c2, r_corner)
        
        # Edge disks
        # Centers at (16, d), (16, 32-d), etc.
        e1 = d_edge
        e2 = size - 1 - d_edge
        
        add_disk(16, e1, r_edge)
        add_disk(16, e2, r_edge)
        add_disk(e1, 16, r_edge)
        add_disk(e2, 16, r_edge)
        
        if invert:
            cell = 1 - cell
            
        # Enforce symmetry (just in case, though construction is symmetric)
        # The construction is symmetric by definition.
        return cell

    # --- 3. Global Optimization ---
    
    def objective(params):
        cell = build_cell(params, invert=False)
        score = evaluate_cell(cell)
        return -score
        
    def objective_inv(params):
        cell = build_cell(params, invert=True)
        score = evaluate_cell(cell)
        return -score

    best_score = -10.0
    best_cell = None
    
    # Bounds: [0, 16] for radii, [0, 16] for distances
    bounds = [(0, 16), (0, 16), (0, 16), (0, 16), (0, 16)]
    
    # Multi-start Nelder-Mead
    n_starts = 10
    
    for i in range(n_starts):
        if time.time() > deadline - 30: break
        
        # Random initial guess
        x0 = np.random.uniform(0, 16, 5)
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'xatol': 0.1, 'fatol': 1e-4, 'maxiter': 200})
            if res.fun < -best_score:
                best_score = -res.fun
                best_cell = build_cell(res.x, invert=False)
        except Exception:
            pass
        
        # Try inverted
        try:
            res_inv = minimize(objective_inv, x0, method='Nelder-Mead', 
                               options={'xatol': 0.1, 'fatol': 1e-4, 'maxiter': 200})
            if res_inv.fun < -best_score:
                best_score = -res_inv.fun
                best_cell = build_cell(res_inv.x, invert=True)
        except Exception:
            pass

    if best_cell is None:
        # Fallback to random
        best_cell = np.random.randint(0, 2, (size, size))
        best_cell = (best_cell + best_cell[::-1, :] + best_cell[:, ::-1] + best_cell[::-1, ::-1]) > 1
        best_score = evaluate_cell(best_cell)

    # --- 4. Local Pixel Refinement ---
    
    current_cell = best_cell.copy()
    current_score = best_score
    
    # Simple Hill Climbing / SA
    temperature = 0.05
    cooling = 0.99
    
    while time.time() < deadline - 5:
        i = np.random.randint(size)
        j = np.random.randint(size)
        
        # Symmetric orbit
        orbit = [
            (i, j),
            (size - 1 - i, j),
            (i, size - 1 - j),
            (size - 1 - i, size - 1 - j),
            (j, i),
            (size - 1 - j, j),
            (j, size - 1 - i),
            (size - 1 - j, size - 1 - i)
        ]
        orbit = list(set(orbit))
        
        new_cell = current_cell.copy()
        for pi, pj in orbit:
            new_cell[pi, pj] = 1 - new_cell[pi, pj]
            
        new_score = evaluate_cell(new_cell)
        
        delta = new_score - current_score
        if delta > 0 or np.random.random() < math.exp(delta / max(temperature, 1e-6)):
            current_cell = new_cell
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_cell = current_cell.copy()
        
        temperature *= cooling
        if temperature < 0.01:
            temperature = 0.01

    return best_cell.tolist()
