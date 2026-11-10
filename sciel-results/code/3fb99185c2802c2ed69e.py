import numpy as np
import time

def design(problem):
    size = problem["pixels"] # 32
    eps_low = problem["eps_low"] # 1
    eps_high = problem["eps_high"] # 12
    
    # 1. Precompute G vectors (289 lowest magnitude)
    max_N = 15
    candidates = []
    for n in range(-max_N, max_N + 1):
        for m in range(-max_N, max_N + 1):
            mag_sq = n*n + m*m
            candidates.append((mag_sq, n, m))
    candidates.sort()
    G_vectors = np.array([(n, m) for _, n, m in candidates[:289]])
    N_pw = len(G_vectors)
    
    # 2. Precompute K points (30 in irreducible BZ)
    k_points = []
    step = np.pi / 10
    for i in range(11):
        kx = i * step
        for j in range(11):
            ky = j * step
            if 0 <= kx <= ky <= np.pi:
                if not any(abs(kx - kp[0]) < 1e-6 and abs(ky - kp[1]) < 1e-6 for kp in k_points):
                    k_points.append((kx, ky))
    if len(k_points) < 30:
         step2 = np.pi / 20
         for i in range(21):
             kx = i * step2
             for j in range(21):
                 ky = j * step2
                 if 0 <= kx <= ky <= np.pi:
                     if not any(abs(kx - kp[0]) < 1e-6 and abs(ky - kp[1]) < 1e-6 for kp in k_points):
                         k_points.append((kx, ky))
                         if len(k_points) >= 30: break
             if len(k_points) >= 30: break
    k_points = np.array(k_points[:30])
    
    # Precompute D and indices for V extraction
    D = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]
    idx_n = D[:, :, 0] % size
    idx_m = D[:, :, 1] % size

    def compute_score(cell_np):
        # cell_np is 32x32 numpy array of 0/1
        # 1 = Dielectric (eps=12), 0 = Air (eps=1)
        # eps_inv = 1/eps
        # Where cell is 1, eps=12, eps_inv=1/12
        # Where cell is 0, eps=1, eps_inv=1
        eps_inv_grid = np.where(cell_np == 1, 1.0/eps_high, 1.0/eps_low)
        
        fourier_coeffs = np.fft.fft2(eps_inv_grid) / (size * size)
        V = fourier_coeffs[idx_n, idx_m]
        
        min_omega_5 = float('inf')
        max_omega_4 = float('-inf')
        
        for kx, ky in k_points:
            k_vec = np.array([kx, ky])
            G_vals = 2 * np.pi * G_vectors
            k_plus_G = k_vec + G_vals
            K_sq = np.sum(k_plus_G**2, axis=1)
            
            M = V * K_sq[np.newaxis, :]
            
            try:
                eigs = np.linalg.eigvals(M)
                real_eigs = np.sort(eigs.real)
                if len(real_eigs) >= 5:
                    w4_sq = real_eigs[3]
                    w5_sq = real_eigs[4]
                    w4 = np.sqrt(max(w4_sq, 0))
                    w5 = np.sqrt(max(w5_sq, 0))
                    
                    max_omega_4 = max(max_omega_4, w4)
                    min_omega_5 = min(min_omega_5, w5)
            except:
                continue
        
        if min_omega_5 == float('inf'):
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        ratio = 2 * gap / (min_omega_5 + max_omega_4)
        return ratio

    start_time = time.time()
    deadline = start_time + 115.0
    
    best_score = -10.0
    best_cell = None
    
    def make_initial_cell(radius):
        cell = np.ones((size, size), dtype=int)
        c = 0.5
        for i in range(size):
            x = (i + 0.5) / size - c
            for j in range(size):
                y = (j + 0.5) / size - c
                if x*x + y*y < radius*radius:
                    cell[i, j] = 0
        return cell

    # Try initial guesses
    for r in [0.15, 0.20, 0.25, 0.30]:
        if time.time() > deadline:
            break
        cell = make_initial_cell(r)
        score = compute_score(cell)
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
            
    if best_cell is None:
        best_cell = make_initial_cell(0.2)
        best_score = compute_score(best_cell)
        
    current_score = best_score
    cell_np = best_cell.copy()
    
    # Hill Climbing
    while time.time() < deadline - 5:
        i = np.random.randint(size)
        j = np.random.randint(size)
        
        old_val = cell_np[i, j]
        cell_np[i, j] = 1 - old_val
        
        new_score = compute_score(cell_np)
        
        if new_score > current_score:
            current_score = new_score
            best_cell = cell_np.copy()
            best_score = new_score
        else:
            cell_np[i, j] = old_val
            
    # Convert back to list of lists
    return best_cell.tolist()
