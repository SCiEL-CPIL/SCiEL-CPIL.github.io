import numpy as np
import time
import math
import random

def design(problem):
    """
    Designs a 32x32 unit cell for a TM band gap between bands 4 and 5.
    Strategy:
    1. Initialize with a symmetric configuration using superellipses.
    2. Use a fast plane-wave evaluator with a moderate number of plane waves.
    3. Enforce C4v symmetry throughout.
    4. Use a multi-start local search with simulated annealing on a parameterized 
       representation (superellipse parameters) to find complex dielectric patterns.
    """
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Time management
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin for 200s limit

    # --- Symmetry Enforcement ---
    def enforce_symmetry(c):
        sym = np.zeros_like(c)
        for i in range(size):
            for j in range(size):
                points = [
                    (i, j),
                    (size - 1 - i, j),
                    (i, size - 1 - j),
                    (size - 1 - i, size - 1 - j),
                    (j, i),
                    (size - 1 - j, j),
                    (j, size - 1 - i),
                    (size - 1 - j, size - 1 - i)
                ]
                s = sum(c[pi, pj] for pi, pj in points)
                sym[i, j] = 1 if s >= 4 else 0
        return sym

    # --- Parameterized Design Space: Superellipses ---
    # A superellipse is defined by |x/a|^n + |y/b|^n <= 1
    # We use 4 superellipses at the quarter-points, with varying parameters.
    # Parameters: [cx, cy, a, b, n] for each superellipse, but by symmetry we
    # only need one set of parameters (a, b, n, cx, cy) and replicate.
    
    def make_cell_from_params(params):
        """
        Create a 32x32 cell from superellipse parameters.
        params: [a, b, n, cx, cy] where (cx, cy) is the center of one superellipse
                in the first quadrant. By symmetry, we replicate to all four quadrants.
        """
        a, b, n, cx, cy = params
        cell = np.zeros((size, size), dtype=int)
        
        # Centers in all four quadrants
        centers = [
            (cy, cx),
            (cy, size - 1 - cx),
            (size - 1 - cy, cx),
            (size - 1 - cy, size - 1 - cx)
        ]
        
        for cy_c, cx_c in centers:
            for i in range(size):
                for j in range(size):
                    dx = abs(i - cy_c) / a if a > 0 else 999
                    dy = abs(j - cx_c) / b if b > 0 else 999
                    if n > 0:
                        val = dx**n + dy**n
                    else:
                        val = 999
                    if val <= 1.0:
                        cell[i, j] = 1
        
        return enforce_symmetry(cell)

    # --- Fast Plane-Wave Evaluator ---
    # Use a moderate number of plane waves for accuracy
    pw_half = 4  # 9x9 = 81 plane waves
    G_vectors = [(h, k) for h in range(-pw_half, pw_half + 1) 
                 for k in range(-pw_half, pw_half + 1)]
    N_pw = len(G_vectors)
    
    Gx_arr = np.array([g[0] for g in G_vectors])
    Gy_arr = np.array([g[1] for g in G_vectors])
    
    # Precompute G differences
    G_diffs = np.zeros((N_pw, N_pw, 2), dtype=int)
    for i in range(N_pw):
        for j in range(N_pw):
            G_diffs[i, j, 0] = G_vectors[i][0] - G_vectors[j][0]
            G_diffs[i, j, 1] = G_vectors[i][1] - G_vectors[j][1]
    
    G_diff_idx_h = G_diffs[:, :, 0] % size
    G_diff_idx_k = G_diffs[:, :, 1] % size
    
    # K-points in irreducible BZ
    k_points = []
    k_points.append((0.0, 0.0))
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        k_points.append((0.0, t))
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        k_points.append((t, 0.5 - t))
    k_points.append((0.5, 0.5))
    
    def evaluate_gap(cell_grid):
        """
        Computes the approximate band gap between bands 4 and 5.
        cell_grid: 32x32 array of 0/1 (0=air, 1=dielectric)
        """
        eps_grid = np.where(cell_grid == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_grid
        
        fft_1eps = np.fft.fft2(inv_eps)
        coeffs = fft_1eps[G_diff_idx_h, G_diff_idx_k] / (size * size)
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            Kx = Gx_arr + kx
            Ky = Gy_arr + ky
            
            dots = np.outer(Kx, Kx) + np.outer(Ky, Ky)
            H = dots * coeffs
            H_real = H.real
            
            # Symmetrize to ensure Hermitian
            H_real = (H_real + H_real.T) / 2.0
            
            eigenvalues = np.linalg.eigvalsh(H_real)
            
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

    # --- Optimization: Multi-start Simulated Annealing on Parameters ---
    best_cell = None
    best_score = -1.0
    
    # Define parameter bounds
    # a, b: semi-axes (0.5 to 12.0)
    # n: superellipse exponent (1 to 10)
    # cx, cy: center in first quadrant (4 to 14)
    param_bounds = [
        (0.5, 12.0),  # a
        (0.5, 12.0),  # b
        (1.0, 10.0),  # n
        (4.0, 14.0),  # cx
        (4.0, 14.0)   # cy
    ]
    
    def random_params():
        return [random.uniform(lo, hi) for lo, hi in param_bounds]
    
    # Try multiple initial configurations
    n_starts = 15
    
    for start_idx in range(n_starts):
        if time.time() > deadline - 5.0:
            break
        
        # Initialize parameters
        if start_idx == 0:
            # Start with a known good configuration (circular disks)
            current_params = [4.0, 4.0, 2.0, 8.0, 8.0]
        else:
            current_params = random_params()
        
        current_cell = make_cell_from_params(current_params)
        current_score = evaluate_gap(current_cell)
        
        if current_score > best_score:
            best_score = current_score
            best_cell = current_cell.copy()
        
        # Simulated annealing
        temperature = 0.3
        cooling_rate = 0.99
        max_iterations = 2000
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 5.0:
                break
            
            # Perturb parameters
            new_params = current_params.copy()
            for i in range(len(new_params)):
                if random.random() < 0.5:
                    new_params[i] += random.gauss(0, 0.5)
                    lo, hi = param_bounds[i]
                    new_params[i] = max(lo, min(hi, new_params[i]))
            
            new_cell = make_cell_from_params(new_params)
            new_score = evaluate_gap(new_cell)
            
            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(delta / max(temperature, 1e-6)):
                current_params = new_params
                current_cell = new_cell
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_cell = current_cell.copy()
            
            temperature *= cooling_rate
            if temperature < 0.01:
                temperature = 0.01
    
    # --- Final Local Search on Pixels ---
    if best_cell is not None and time.time() < deadline - 10.0:
        # Do a quick pixel-level local search to refine
        current_cell = best_cell.copy()
        current_score = best_score
        
        for iteration in range(500):
            if time.time() > deadline - 5.0:
                break
            
            i = random.randint(0, size - 1)
            j = random.randint(0, size - 1)
            
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
            
            new_score = evaluate_gap(new_cell)
            
            if new_score > current_score:
                current_cell = new_cell
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_cell = current_cell.copy()
    
    if best_cell is None:
        # Fallback: simple disk configuration
        best_cell = np.zeros((size, size), dtype=int)
        centers = [(8, 8), (8, 24), (24, 8), (24, 24)]
        radius = 4.0
        for cy, cx in centers:
            for i in range(size):
                for j in range(size):
                    dist = math.sqrt((i - cy)**2 + (j - cx)**2)
                    if dist < radius:
                        best_cell[i, j] = 1
        best_cell = enforce_symmetry(best_cell)
    
    return best_cell.tolist()
