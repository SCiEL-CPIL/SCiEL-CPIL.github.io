import numpy as np
import time

def design(problem):
    start_time = time.time()
    deadline = 110.0  # Safety margin from 120s limit
    
    size = problem["pixels"]
    eps_high = problem["eps_high"]
    eps_low = problem["eps_low"]
    
    # 1. Define the fundamental domain
    fd_indices = []
    for i in range(16):
        for j in range(i, 16):
            fd_indices.append((i, j))
            
    n_fd = len(fd_indices)
    
    def build_cell(fd_values):
        cell = np.zeros((size, size), dtype=np.int8)
        for k, (i, j) in enumerate(fd_indices):
            val = fd_values[k]
            positions = [
                (i, j),
                (j, i),
                (i, 31-j),
                (31-j, i),
                (31-i, j),
                (j, 31-i),
                (31-i, 31-j),
                (31-j, 31-i)
            ]
            for (r, c) in positions:
                cell[r, c] = val
        return cell

    # Fast Objective Function (Vectorized)
    N_w = 9 # Number of G vectors in one dimension (-4 to 4)
    g_range = np.arange(-N_w//2, N_w//2 + 1)
    g_vectors = [(m, n) for m in g_range for n in g_range]
    num_g = len(g_vectors)
    
    # Integer G vectors
    G_int = np.array(g_vectors, dtype=float)
    
    # K-points in units of B = 2pi/size
    # Irreducible BZ: 0 <= ky <= kx <= 0.5 (in units of B)
    k_points = [
        (0.0, 0.0),      # Gamma
        (0.5, 0.0),    # X
        (0.5, 0.5),  # M
        (0.25, 0.0),
        (0.25, 0.25),
        (0.4, 0.0),
        (0.0, 0.25),
        (0.2, 0.2),
        (0.3, 0.3),
        (0.4, 0.1),
    ]
    
    # Precompute FFT indices for G vectors
    g_fft_indices = [(m % size, n % size) for m, n in g_vectors]
    
    # Precompute the difference indices for chi matrix
    m_arr = np.array([g[0] for g in g_vectors])
    n_arr = np.array([g[1] for g in g_vectors])
    
    m_diff = m_arr[:, np.newaxis] - m_arr[np.newaxis, :]
    n_diff = n_arr[:, np.newaxis] - n_arr[np.newaxis, :]
    
    fft_idx_i = m_diff % size
    fft_idx_j = n_diff % size
    
    B = 2.0 * np.pi / size
    B_sq = B * B
    
    def evaluate_gap(cell_2d):
        chi_low = 1.0 / eps_low
        chi_high = 1.0 / eps_high
        
        chi_map = chi_low + (chi_high - chi_low) * cell_2d.astype(float)
        fft_chi = np.fft.fftn(chi_map)
        
        chi_matrix = fft_chi[fft_idx_i, fft_idx_j]
        chi_matrix = chi_matrix / (size * size)
        
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            K_scaled = np.array([kx, ky])
            
            G_K_scaled = G_int + K_scaled
            
            D_scaled = G_K_scaled @ G_K_scaled.T
            
            H = B_sq * D_scaled * chi_matrix
            
            try:
                eigvals = np.linalg.eigvalsh(H)
                eigvals = eigvals[eigvals > 0]
                if len(eigvals) >= 5:
                    omega_4 = np.sqrt(eigvals[3])
                    omega_5 = np.sqrt(eigvals[4])
                    
                    max_omega_4 = max(max_omega_4, omega_4)
                    min_omega_5 = min(min_omega_5, omega_5)
            except Exception:
                pass
                
        if min_omega_5 == float('inf'):
            return -1.0
            
        gap = 2 * (min_omega_5 - max_omega_4)
        midgap = min_omega_5 + max_omega_4
        ratio = gap / midgap if midgap > 0 else 0
        return ratio

    # Search Loop
    best_score = -1.0
    best_fd = None
    
    def make_fd_from_pattern(pattern_func):
        cell = np.ones((size, size), dtype=np.int8)
        pattern_func(cell)
        fd_vals = []
        for i, j in fd_indices:
            fd_vals.append(cell[i, j])
        return fd_vals

    def pattern_central_circle(cell, radius=4.0):
        i, j = np.indices((size, size))
        dist = np.sqrt((i - 15.5)**2 + (j - 15.5)**2)
        cell[dist < radius] = 0

    def pattern_central_square(cell, half_width=4.0):
        i, j = np.indices((size, size))
        dist_x = np.abs(i - 15.5)
        dist_y = np.abs(j - 15.5)
        mask = (dist_x < half_width) & (dist_y < half_width)
        cell[mask] = 0

    def pattern_4_circles(cell, radius=3.0):
        i, j = np.indices((size, size))
        dist1 = np.sqrt((i - 8.5)**2 + (j - 8.5)**2)
        dist2 = np.sqrt((i - 8.5)**2 + (j - 23.5)**2)
        dist3 = np.sqrt((i - 23.5)**2 + (j - 8.5)**2)
        dist4 = np.sqrt((i - 23.5)**2 + (j - 23.5)**2)
        mask = (dist1 < radius) | (dist2 < radius) | (dist3 < radius) | (dist4 < radius)
        cell[mask] = 0

    def pattern_4_squares(cell, half_width=2.5):
        i, j = np.indices((size, size))
        centers = [(8.5, 8.5), (8.5, 23.5), (23.5, 8.5), (23.5, 23.5)]
        mask = np.zeros((size, size), dtype=bool)
        for cx, cy in centers:
            dx = np.abs(i - cx)
            dy = np.abs(j - cy)
            mask |= (dx < half_width) & (dy < half_width)
        cell[mask] = 0

    initial_configs = [
        make_fd_from_pattern(lambda c: pattern_central_circle(c, 4.0)),
        make_fd_from_pattern(lambda c: pattern_central_square(c, 4.0)),
        make_fd_from_pattern(lambda c: pattern_4_circles(c, 3.0)),
        make_fd_from_pattern(lambda c: pattern_4_squares(c, 2.5)),
        [1 if np.random.rand() > 0.3 else 0 for _ in range(n_fd)]
    ]
    
    for init_fd in initial_configs:
        current_fd = init_fd[:]
        current_score = evaluate_gap(build_cell(current_fd))
        
        if current_score > best_score:
            best_score = current_score
            best_fd = current_fd[:]
            
        for step in range(1000):
            if time.time() > deadline:
                break
                
            idx = np.random.randint(n_fd)
            current_fd[idx] = 1 - current_fd[idx]
            
            new_cell = build_cell(current_fd)
            new_score = evaluate_gap(new_cell)
            
            if new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_fd = current_fd[:]
            else:
                current_fd[idx] = 1 - current_fd[idx]
                
            if time.time() > deadline:
                break

    if best_fd is None:
        fallback_fd = make_fd_from_pattern(lambda c: pattern_central_circle(c, 4.0))
        return build_cell(fallback_fd).tolist()
        
    return build_cell(best_fd).tolist()
