import numpy as np
import time
import math

# EVOLVE-BLOCK-START
def design(problem):
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 195.0

    # Plane wave setup: 9x9 = 81 modes
    pw_half = 4
    G_vectors = [(h, k) for h in range(-pw_half, pw_half + 1) 
                 for k in range(-pw_half, pw_half + 1)]
    N_pw = len(G_vectors)
    
    Gx_arr = np.array([g[0] for g in G_vectors], dtype=float)
    Gy_arr = np.array([g[1] for g in G_vectors], dtype=float)
    
    G_diffs = np.zeros((N_pw, N_pw, 2), dtype=int)
    for i in range(N_pw):
        for j in range(N_pw):
            G_diffs[i, j, 0] = G_vectors[i][0] - G_vectors[j][0]
            G_diffs[i, j, 1] = G_vectors[i][1] - G_vectors[j][1]
            
    G_diff_idx_h = G_diffs[:, :, 0] % size
    G_diff_idx_k = G_diffs[:, :, 1] % size

    # K-points: Gamma, M, X and midpoints
    k_points = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.0, 0.25),
        (0.25, 0.25)
    ]

    def evaluate_gap(cell_grid):
        eps_grid = np.where(cell_grid == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_grid
        
        fft_1eps = np.fft.fft2(inv_eps)
        coeffs = fft_1eps[G_diff_idx_h, G_diff_idx_k] / (size * size)
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            Kx = Gx_arr + kx
            Ky = Gy_arr + ky
            K_mag = np.sqrt(Kx**2 + Ky**2)
            
            H_sym = np.outer(K_mag, K_mag) * coeffs.real
            H_sym = (H_sym + H_sym.T) / 2.0
            
            eigenvalues = np.linalg.eigvalsh(H_sym)
            
            if len(eigenvalues) >= 5:
                w4_sq = eigenvalues[3]
                w5_sq = eigenvalues[4]
                
                if w4_sq > 0 and w5_sq > 0:
                    w4 = math.sqrt(w4_sq)
                    w5 = math.sqrt(w5_sq)
                    
                    if w4 > max_omega_4:
                        max_omega_4 = w4
                    if w5 < min_omega_5:
                        min_omega_5 = w5
        
        if min_omega_5 == float('inf') or max_omega_4 == 0:
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        ratio = 2 * gap / midgap
        return ratio

    def create_cell_from_params(a, b, n):
        cell = np.zeros((size, size), dtype=int)
        centers = [(8, 8), (8, 24), (24, 8), (24, 24)]
        
        for cy, cx in centers:
            for i in range(size):
                for j in range(size):
                    dx = abs(i - cy)
                    dy = abs(j - cx)
                    if a > 0.1 and b > 0.1 and n > 0.1:
                        val = (dx/a)**n + (dy/b)**n
                        if val <= 1.0:
                            cell[i, j] = 1
        return cell

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

    best_score = -1.0
    best_cell = np.zeros((size, size), dtype=int)
    
    a_range = (1.0, 12.0)
    b_range = (1.0, 12.0)
    n_range = (1.0, 10.0)
    
    num_restarts = 100
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        a = np.random.uniform(a_range[0], a_range[1])
        b = np.random.uniform(b_range[0], b_range[1])
        n = np.random.uniform(n_range[0], n_range[1])
        
        current_cell = create_cell_from_params(a, b, n)
        current_cell = enforce_symmetry(current_cell)
        current_score = evaluate_gap(current_cell)
        
        if current_score > best_score:
            best_score = current_score
            best_cell = current_cell.copy()
        
        steps = 100
        for step in range(steps):
            if time.time() > deadline - 5:
                break
                
            da = np.random.normal(0, 0.5)
            db = np.random.normal(0, 0.5)
            dn = np.random.normal(0, 0.5)
            
            new_a = np.clip(a + da, a_range[0], a_range[1])
            new_b = np.clip(b + db, b_range[0], b_range[1])
            new_n = np.clip(n + dn, n_range[0], n_range[1])
            
            new_cell = create_cell_from_params(new_a, new_b, new_n)
            new_cell = enforce_symmetry(new_cell)
            new_score = evaluate_gap(new_cell)
            
            if new_score >= current_score:
                a, b, n = new_a, new_b, new_n
                current_cell = new_cell
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_cell = current_cell.copy()
                    
    # Refine with pixel flips if time remains
    if time.time() < deadline - 20:
        current_cell = best_cell.copy()
        current_score = best_score
        temperature = 0.05
        for _ in range(1000):
            if time.time() > deadline - 10:
                break
            i = np.random.randint(size)
            j = np.random.randint(size)
            orbit = list(set([
                (i, j),
                (size - 1 - i, j),
                (i, size - 1 - j),
                (size - 1 - i, size - 1 - j),
                (j, i),
                (size - 1 - j, j),
                (j, size - 1 - i),
                (size - 1 - j, size - 1 - i)
            ]))
            
            new_cell = current_cell.copy()
            for pi, pj in orbit:
                new_cell[pi, pj] = 1 - new_cell[pi, pj]
            
            new_score = evaluate_gap(new_cell)
            delta = new_score - current_score
            if delta > 0 or np.random.random() < math.exp(delta / max(temperature, 1e-6)):
                current_cell = new_cell
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_cell = current_cell.copy()
            temperature *= 0.99

    final_cell = enforce_symmetry(best_cell)
    return final_cell.tolist()

# EVOLVE-BLOCK-END
