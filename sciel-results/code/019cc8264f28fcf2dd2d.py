import numpy as np
import time

def design(problem):
    start_time = time.time()
    deadline = start_time + 115.0
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # --- Fast Solver Parameters ---
    g_range_fast = range(-5, 6)
    G_vectors_fast = np.array([(gx, gy) for gx in g_range_fast for gy in g_range_fast], dtype=float)
    G_vectors_fast *= 2.0 * np.pi
    N_g_fast = len(G_vectors_fast)
    
    k_points_fast = np.array([
        [0, 0],
        [np.pi/2, 0],
        [np.pi, 0],
        [np.pi/2, np.pi/2],
        [np.pi, np.pi]
    ])

    # --- Slow Solver Parameters (for final check) ---
    g_range_slow = range(-8, 9)
    G_vectors_slow = np.array([(gx, gy) for gx in g_range_slow for gy in g_range_slow], dtype=float)
    G_vectors_slow *= 2.0 * np.pi
    N_g_slow = len(G_vectors_slow)
    
    k_points_slow = []
    n_k = 10
    for i in range(n_k):
        for j in range(i, n_k):
            k_points_slow.append([i / (n_k - 1) * np.pi, j / (n_k - 1) * np.pi])
    k_points_slow = np.array(k_points_slow)

    def compute_epsilon_fourier(cell, g_range):
        eps_grid = np.where(np.array(cell) == 1, eps_high, eps_low)
        eps_fft = np.fft.fft2(eps_grid)
        eps_fft /= (size * size)
        
        eps_G = {}
        for gx in g_range:
            for gy in g_range:
                ix = gx % size
                iy = gy % size
                eps_G[(gx, gy)] = eps_fft[iy, ix]
        return eps_G

    def solve_tm_bands(eps_G, G_vectors, kx, ky, g_range):
        N = len(G_vectors)
        K = np.zeros((2*N, 2*N), dtype=complex)
        E = np.zeros((2*N, 2*N), dtype=complex)
        
        kG_x = kx + G_vectors[:, 0]
        kG_y = ky + G_vectors[:, 1]
        
        for i in range(N):
            kx_i = kG_x[i]
            ky_i = kG_y[i]
            K[2*i, 2*i] = ky_i**2
            K[2*i, 2*i+1] = -kx_i * ky_i
            K[2*i+1, 2*i] = -ky_i * kx_i
            K[2*i+1, 2*i+1] = kx_i**2
            
        for i in range(N):
            for j in range(N):
                dGx = int(round(G_vectors[i, 0] - G_vectors[j, 0])) // (2*np.pi)
                dGy = int(round(G_vectors[i, 1] - G_vectors[j, 1])) // (2*np.pi)
                # Note: G_vectors are multiplied by 2pi. So difference is 2pi * n.
                # We need the index n.
                # dGx_calculated = G_vectors[i,0] - G_vectors[j,0]
                # n_x = dGx_calculated / (2*pi)
                eps_val = eps_G[(dGx, dGy)]
                E[2*i, 2*j] = eps_val
                E[2*i, 2*j+1] = eps_val
                E[2*i+1, 2*j] = eps_val
                E[2*i+1, 2*j+1] = eps_val
                
        try:
            eigvals, _ = np.linalg.eigh(K, E)
            omegas = np.sqrt(np.real(eigvals))
            omegas.sort()
            return omegas
        except:
            return None

    def get_score_fast(cell):
        if time.time() > deadline:
            return -1.0
        g_range = range(-11, 12) # Max diff is 10
        eps_G = compute_epsilon_fourier(cell, g_range)
        max_w4 = 0.0
        min_w5 = float('inf')
        
        for kx, ky in k_points_fast:
            omegas = solve_tm_bands(eps_G, G_vectors_fast, kx, ky, g_range)
            if omegas is None or len(omegas) < 5:
                continue
            w4 = omegas[3]
            w5 = omegas[4]
            if w4 > max_w4:
                max_w4 = w4
            if w5 < min_w5:
                min_w5 = w5
                
        if min_w5 == float('inf'):
            return -1.0
        gap = min_w5 - max_w4
        midgap = (min_w5 + max_w4) / 2.0
        if midgap == 0:
            return -1.0
        return 2.0 * gap / midgap

    def get_score_slow(cell):
        if time.time() > deadline:
            return -1.0
        g_range = range(-17, 18) # Max diff is 16
        eps_G = compute_epsilon_fourier(cell, g_range)
        max_w4 = 0.0
        min_w5 = float('inf')
        
        for kx, ky in k_points_slow:
            omegas = solve_tm_bands(eps_G, G_vectors_slow, kx, ky, g_range)
            if omegas is None or len(omegas) < 5:
                continue
            w4 = omegas[3]
            w5 = omegas[4]
            if w4 > max_w4:
                max_w4 = w4
            if w5 < min_w5:
                min_w5 = w5
                
        if min_w5 == float('inf'):
            return -1.0
        gap = min_w5 - max_w4
        midgap = (min_w5 + max_w4) / 2.0
        if midgap == 0:
            return -1.0
        return 2.0 * gap / midgap

    def make_cell_4_rods(d, r):
        cell = [[0]*size for _ in range(size)]
        cx1 = 15.5 - d
        cy1 = 15.5 - d
        cx2 = 15.5 + d
        cy2 = 15.5 + d
        centers = [(cx1, cy1), (cx2, cy1), (cx1, cy2), (cx2, cy2)]
        for i in range(size):
            for j in range(size):
                px = i + 0.5
                py = j + 0.5
                for cx, cy in centers:
                    if (px - cx)**2 + (py - cy)**2 < r**2:
                        cell[i][j] = 1
                        break
        return cell

    # Search
    best_cell = make_cell_4_rods(7.5, 2.5)
    best_score = get_score_fast(best_cell)
    
    d_cur = 7.5
    r_cur = 2.5
    step_d = 1.0
    step_r = 0.5
    
    while time.time() < deadline - 20.0: # Keep 20s for final check
        d_new = d_cur + (np.random.rand() - 0.5) * 2 * step_d
        r_new = r_cur + (np.random.rand() - 0.5) * 2 * step_r
        
        if d_new < 1.0 or d_new > 15.0 or r_new < 0.5 or r_new > 15.5 - d_new:
            continue
            
        cell_new = make_cell_4_rods(d_new, r_new)
        score_new = get_score_fast(cell_new)
        
        if score_new > best_score:
            best_score = score_new
            best_cell = cell_new
            d_cur = d_new
            r_cur = r_new
            step_d *= 0.8
            step_r *= 0.8
        else:
            # No improvement, maybe reduce steps or restart?
            pass

    # Final verification with slow solver
    final_score = get_score_slow(best_cell)
    
    # If slow score is bad, maybe try a few other candidates?
    # For now, return the best found.
    
    return best_cell
