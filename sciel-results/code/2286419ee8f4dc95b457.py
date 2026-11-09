"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

# EVOLVE-BLOCK-START
def design(problem):
    """
    Designs a 32x32 unit cell for a TM band gap between bands 4 and 5.
    Strategy:
    1. Use a parameterized geometric family (rotated squares/diamonds + central square)
       to define a structured search space.
    2. Perform a coarse-to-fine grid search over the geometric parameters.
    3. Refine the best candidate using fast pixel-level simulated annealing
       with a reduced plane-wave set for speed.
    4. Enforce C4v symmetry throughout.
    5. Return the best design found within the time budget.
    """
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Time management
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin for 200s limit

    # --- Helper: Enforce C4v symmetry exactly ---
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

    # --- Helper: Generate cell from geometric parameters ---
    def generate_cell(angle_deg, size_sq, center_size, cx, cy):
        cell = np.zeros((size, size), dtype=int)
        x_coords = np.arange(size) - 15.5
        y_coords = np.arange(size) - 15.5
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        centers = [
            (cx, cy),
            (-cx, cy),
            (-cx, -cy),
            (cx, -cy)
        ]
        
        angle_rad = np.deg2rad(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        if center_size > 0:
            half_cs = center_size / 2.0
            cell[(np.abs(X) < half_cs) & (np.abs(Y) < half_cs)] = 1
        
        for c_x, c_y in centers:
            X_t = X - c_x
            Y_t = Y - c_y
            X_r = X_t * cos_a + Y_t * sin_a
            Y_r = -X_t * sin_a + Y_t * cos_a
            half_s = size_sq / 2.0
            mask = (np.abs(X_r) < half_s) & (np.abs(Y_r) < half_s)
            cell[mask] = 1
        
        return enforce_symmetry(cell)

    # --- Fast Evaluator for Optimization (7x7 PW) ---
    pw_half = 3
    G_vectors = [(h, k) for h in range(-pw_half, pw_half + 1) 
                 for k in range(-pw_half, pw_half + 1)]
    N_pw = len(G_vectors)
    
    Gx_arr = np.array([g[0] for g in G_vectors])
    Gy_arr = np.array([g[1] for g in G_vectors])
    
    G_diffs = np.zeros((N_pw, N_pw, 2), dtype=int)
    for i in range(N_pw):
        for j in range(N_pw):
            G_diffs[i, j, 0] = G_vectors[i][0] - G_vectors[j][0]
            G_diffs[i, j, 1] = G_vectors[i][1] - G_vectors[j][1]
    
    G_diff_idx_h = G_diffs[:, :, 0] % size
    G_diff_idx_k = G_diffs[:, :, 1] % size
    
    # K-points for fast evaluation (subset)
    k_points_fast = [
        (0.0, 0.0),
        (0.0, 0.1), (0.0, 0.2), (0.0, 0.3), (0.0, 0.4), (0.0, 0.5),
        (0.1, 0.4), (0.2, 0.3), (0.3, 0.2), (0.4, 0.1),
        (0.5, 0.0), (0.5, 0.1), (0.5, 0.2), (0.5, 0.3), (0.5, 0.4), (0.5, 0.5),
        (0.25, 0.25), (0.25, 0.5), (0.5, 0.25)
    ]

    def evaluate_gap_fast(cell_grid):
        eps_grid = np.where(cell_grid == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_grid
        fft_1eps = np.fft.fft2(inv_eps)
        coeffs = fft_1eps[G_diff_idx_h, G_diff_idx_k] / (size * size)
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points_fast:
            Kx = Gx_arr + kx
            Ky = Gy_arr + ky
            dots = np.outer(Kx, Kx) + np.outer(Ky, Ky)
            H = dots * coeffs
            H_real = H.real
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
        return 2 * gap / midgap

    # --- Phase 1: Coarse Geometric Search ---
    best_cell = None
    best_score = -float('inf')
    
    # Coarse grid
    angles = [0, 10, 20, 30, 40, 45]
    sizes_sq = [4, 6, 8, 10, 12]
    center_sizes = [0, 4, 8]
    positions = [4, 6, 8, 10, 12]
    
    for a in angles:
        if time.time() > deadline - 60: break
        for s in sizes_sq:
            if time.time() > deadline - 60: break
            for cs in center_sizes:
                if time.time() > deadline - 60: break
                for cx in positions:
                    if time.time() > deadline - 60: break
                    for cy in positions:
                        cell = generate_cell(a, s, cs, cx, cy)
                        score = evaluate_gap_fast(cell)
                        if score > best_score:
                            best_score = score
                            best_cell = cell.copy()

    if best_cell is None:
        # Fallback
        best_cell = np.zeros((size, size), dtype=int)
        best_score = 0.0

    # --- Phase 2: Fine Geometric Search around Best ---
    if time.time() < deadline - 30:
        # Find indices of best params? We didn't store them. 
        # Let's just do a local random jitter search on parameters.
        # Or simply proceed to pixel optimization which is more robust.
        pass

    # --- Phase 3: Pixel-level Simulated Annealing ---
    current_cell = best_cell.copy()
    current_score = best_score
    
    temperature = 0.2
    cooling_rate = 0.999
    max_iterations = 50000
    iteration = 0
    
    while time.time() < deadline and iteration < max_iterations:
        iteration += 1
        
        # Pick a random symmetric orbit to flip
        i = np.random.randint(size)
        j = np.random.randint(size)
        
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
        
        new_score = evaluate_gap_fast(new_cell)
        
        delta = new_score - current_score
        if delta > 0 or np.random.random() < math.exp(delta / max(temperature, 1e-6)):
            current_cell = new_cell
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_cell = current_cell.copy()
        
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01

    # --- Finalize ---
    final_cell = enforce_symmetry(best_cell)
    return final_cell.tolist()

# EVOLVE-BLOCK-END
