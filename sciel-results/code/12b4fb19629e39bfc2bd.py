import numpy as np
import time
import scipy.linalg

def design(problem):
    start_time = time.time()
    deadline = start_time + 115.0 

    N = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # PWE Parameters
    N_pw = 15 # 15x15 = 225 plane waves
    G_range = np.arange(-N_pw//2, N_pw//2 + 1)
    G_vectors = []
    for gx in G_range:
        for gy in G_range:
            G_vectors.append((2 * np.pi * gx / N, 2 * np.pi * gy / N))
    G_vectors = np.array(G_vectors)
    n_pw = len(G_vectors)
    
    # K points
    k_points = [
        np.array([0.0, 0.0]),
        np.array([0.5, 0.0]),
        np.array([0.5, 0.5])
    ]
    k_cartesian = [2 * np.pi * k for k in k_points]

    def expand_quadrant(q):
        cell = np.zeros((N, N), dtype=int)
        for i in range(8):
            for j in range(8):
                val = q[i][j]
                cell[i][j] = val
                cell[i][N-1-j] = val
                cell[N-1-i][j] = val
                cell[N-1-i][N-1-j] = val
        return cell

    def compute_eps_coeffs(cell):
        eps_grid = np.where(cell == 1, eps_high, eps_low)
        coeffs = np.zeros(n_pw, dtype=complex)
        i_idx, j_idx = np.indices((N, N))
        for idx, (Gx, Gy) in enumerate(G_vectors):
            phase = -1j * (Gx * (i_idx + 0.5) / N + Gy * (j_idx + 0.5) / N)
            coeffs[idx] = np.sum(eps_grid * np.exp(phase)) / (N * N)
        return coeffs

    def solve_bands(cell):
        eps_coeffs = compute_eps_coeffs(cell)
        eps_dict = {tuple(np.round(G_vectors[i], decimals=9)): eps_coeffs[i] for i in range(n_pw)}
        
        min_omega_5 = np.inf
        max_omega_4 = 0.0
        
        for k in k_cartesian:
            k_plus_G = k + G_vectors
            k_sq = np.sum(k_plus_G**2, axis=1)
            A = np.diag(k_sq)
            
            B = np.zeros((n_pw, n_pw), dtype=complex)
            for i in range(n_pw):
                for j in range(n_pw):
                    G_diff = G_vectors[i] - G_vectors[j]
                    key = tuple(np.round(G_diff, decimals=9))
                    if key in eps_dict:
                        B[i, j] = eps_dict[key]
            
            try:
                eigvals, _ = scipy.linalg.eig(A, B)
                real_eigs = np.real(eigvals)
                valid = real_eigs > 1e-6
                real_eigs = np.sort(real_eigs[valid])
                
                if len(real_eigs) >= 5:
                    w4 = np.sqrt(real_eigs[3])
                    w5 = np.sqrt(real_eigs[4])
                    if w4 > max_omega_4:
                        max_omega_4 = w4
                    if w5 < min_omega_5:
                        min_omega_5 = w5
            except Exception:
                continue
        
        if max_omega_4 == 0 or min_omega_5 == np.inf:
            return -1.0
        return 2.0 * (min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)

    # Initial Candidates
    candidates = []
    
    def make_shape(shape_type, param1, param2=None):
        q = np.zeros((8, 8), dtype=int)
        for i in range(8):
            for j in range(8):
                x = (i + 0.5) / N - 0.5
                y = (j + 0.5) / N - 0.5
                if shape_type == 'circle':
                    if x*x + y*y < param1*param1:
                        q[i][j] = 1
                elif shape_type == 'square':
                    if abs(x) < param1 and abs(y) < param1:
                        q[i][j] = 1
                elif shape_type == 'superellipse':
                    n_exp = param1
                    size = param2
                    if (abs(x)/size)**n_exp + (abs(y)/size)**n_exp < 1.0:
                        q[i][j] = 1
        # Enforce symmetry q[i][j] = q[j][i]
        for i in range(8):
            for j in range(i+1, 8):
                q[j][i] = q[i][j]
        return q

    candidates.append(make_shape('circle', 0.2))
    candidates.append(make_shape('square', 0.2))
    for exp in [3, 4, 5, 6]:
        for size in [0.15, 0.2, 0.25, 0.3]:
            candidates.append(make_shape('superellipse', exp, size))

    best_score = -10.0
    best_q = None
    
    for q in candidates:
        if time.time() > deadline: break
        cell = expand_quadrant(q)
        score = solve_bands(cell)
        if score > best_score:
            best_score = score
            best_q = q.copy()

    current_q = best_q.copy()
    current_score = best_score
    no_improve_count = 0
    
    while time.time() < deadline:
        if no_improve_count > 100:
            break
            
        new_q = current_q.copy()
        num_mutations = np.random.randint(1, 4)
        for _ in range(num_mutations):
            ri = np.random.randint(0, 8)
            rj = np.random.randint(0, 8)
            new_q[ri][rj] = 1 - new_q[ri][rj]
            new_q[rj][ri] = 1 - new_q[rj][ri]
            
        cell = expand_quadrant(new_q)
        score = solve_bands(cell)
        
        if score > current_score:
            current_q = new_q
            current_score = score
            no_improve_count = 0
            if score > best_score:
                best_score = score
                best_q = current_q.copy()
        else:
            no_improve_count += 1
            
    final_cell = expand_quadrant(best_q)
    return final_cell.tolist()
