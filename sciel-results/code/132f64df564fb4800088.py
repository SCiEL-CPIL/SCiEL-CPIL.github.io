import numpy as np
import time
import math

def design(problem):
    N = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    lower_band = problem["lower_band"] # 4
    
    start_time = time.time()
    time_limit = 220 # Leave margin
    
    # Symmetry
    def get_orbit(i, j, N):
        return [
            (i, j), (j, i),
            (N-1-i, j), (N-1-i, N-1-j),
            (N-1-j, N-1-i), (N-1-j, i),
            (i, N-1-j), (j, N-1-i)
        ]
    
    def get_free_pixels(N):
        free = []
        used = set()
        for i in range(N):
            for j in range(N):
                if (i,j) in used: continue
                free.append((i,j))
                for p in set(get_orbit(i, j, N)):
                    used.add(p)
        return free
    
    free_pixels = get_free_pixels(N)
    n_free = len(free_pixels)
    
    # K-points (30 points in irreducible BZ)
    # Boundary: kx=ky (Gamma-X) and kx=0 (M-X) ?
    # Irreducible wedge: 0 <= ky <= kx, kx <= pi, ky <= pi.
    # Actually, square BZ: -pi..pi.
    # Irreducible: 0 <= ky <= kx <= pi.
    # Points:
    k_points = []
    # Gamma-X: (k, 0)
    for k in np.linspace(0, np.pi, 10):
        k_points.append((k, 0))
    # X-M: (pi, k)
    for k in np.linspace(0, np.pi, 10):
        k_points.append((np.pi, k))
    # M-Gamma: (k, k)
    for k in np.linspace(0, np.pi, 10):
        k_points.append((k, k))
    # Remove duplicates if any (Gamma, X, M)
    # This gives 30 points.
    
    def build_solver(n_g):
        # n_g: half-range, e.g., 4 for 9x9, 8 for 17x17
        # G vectors
        nx = np.arange(-n_g, n_g+1)
        G_x = 2 * np.pi * nx
        G_y = G_x
        G_x_grid, G_y_grid = np.meshgrid(G_x, G_y, indexing='ij')
        G_flat_x = G_x_grid.flatten()
        G_flat_y = G_y_grid.flatten()
        n_pw = len(G_flat_x)
        
        # Q vectors (differences)
        n_q = 2 * n_g + 1
        nqx = np.arange(-2*n_g, 2*n_g+1)
        Q_x = 2 * np.pi * nqx
        Q_y = Q_x
        Q_x_grid, Q_y_grid = np.meshgrid(Q_x, Q_y, indexing='ij')
        Q_flat_x = Q_x_grid.flatten()
        Q_flat_y = Q_y_grid.flatten()
        n_q_total = len(Q_flat_x)
        
        # Precompute Exponentials for Q
        X = (np.arange(N) + 0.5) / N - 0.5
        Y = X
        # ExpTensor shape: (n_q_total, N, N)
        # Q_flat_x[:, None, None] * X[None, :, None]
        ExpTensor = np.exp(-1j * (Q_flat_x[:, None, None] * X[None, :, None] + 
                                  Q_flat_y[:, None, None] * Y[None, None, :]))
        
        # Map Q index
        # Q index = (nqx - (-2*n_g)) * (4*n_g+1) + (nqy - (-2*n_g))
        # Or just use a 2D array for eps_Q
        
        def score(cell):
            eps_2d = np.where(np.array(cell) == 1, eps_high, eps_low)
            # eps_Q = sum(eps_2d * ExpTensor) / N^2
            eps_Q = np.einsum('ij,qij->q', eps_2d, ExpTensor) / (N * N)
            
            # Reshape eps_Q to 2D for easy indexing
            # n_q x n_q
            eps_Q_2d = eps_Q.reshape((2*n_g+1)*2, 2*n_g+1) # Wait, size is 4*n_g+1?
            # n_q = 2*n_g + 1? No.
            # G range: -n_g .. n_g. Size 2*n_g+1.
            # Q range: -2*n_g .. 2*n_g. Size 4*n_g+1.
            # My n_q variable above was confusing.
            # Let's fix:
            # G size: 2*n_g + 1.
            # Q size: 4*n_g + 1.
            # n_q_total = (4*n_g+1)^2.
            # eps_Q_2d shape: (4*n_g+1, 4*n_g+1).
            
            # Corrected Q grid construction
            # n_q_side = 4*n_g + 1
            eps_Q_2d = eps_Q.reshape((4*n_g+1, 4*n_g+1))
            
            # G indices for M
            # G_flat_x, G_flat_y are length n_pw
            # We need E[m, n] = eps_Q_2d[ (G_x[m] - G_x[n])/2pi + 2*n_g, (G_y[m] - G_y[n])/2pi + 2*n_g ]
            # Let's precompute the indices
            G_idx_x = np.arange(-n_g, n_g+1) # Values
            # G_flat_x is just 2*pi * G_idx_x repeated.
            # So difference in indices is just G_idx_x[m] - G_idx_x[n].
            
            # Create index matrix
            # G_idx_flat_x = G_idx_x.repeat(2*n_g+1) ? No.
            # G_x_grid, G_y_grid were meshgrid.
            # G_flat_x corresponds to G_x_grid.flatten().
            # G_x_grid[i, j] = G_x[i].
            # So G_flat_x[k] = G_x[i] where k = i*(2*n_g+1) + j.
            
            # Let's just compute the difference matrix directly
            # G_x_vals = G_flat_x / (2*pi)
            # G_y_vals = G_flat_y / (2*pi)
            
            G_x_vals = G_flat_x / (2 * np.pi)
            G_y_vals = G_flat_y / (2 * np.pi)
            
            diff_x = G_x_vals[:, None] - G_x_vals[None, :]
            diff_y = G_y_vals[:, None] - G_y_vals[None, :]
            
            # Indices into eps_Q_2d
            idx_x = (diff_x + 2 * n_g).astype(int)
            idx_y = (diff_y + 2 * n_g).astype(int)
            
            E = eps_Q_2d[idx_x, idx_y]
            
            max_gap = -1
            for kx, ky in k_points:
                K_x = kx + G_flat_x
                K_y = ky + G_flat_y
                
                A = np.outer(K_x, K_x)
                B = np.outer(K_y, K_y)
                M_base = A + B
                M = E * M_base
                
                eigvals = np.linalg.eigvalsh(M)
                eigvals = np.sort(eigvals)
                
                w4 = eigvals[lower_band - 1]
                w5 = eigvals[lower_band]
                
                if w5 + w4 == 0: continue
                gap = 2 * (w5 - w4) / (w5 + w4)
                if gap > max_gap:
                    max_gap = gap
                    
            return max_gap
        
        return score

    score_coarse = build_solver(4) # 9x9
    score_full = build_solver(8)   # 17x17
    
    # Optimization
    best_cell = None
    best_score_full = -1
    best_score_coarse = -1
    
    # Initial guesses
    inits = []
    # 1. Rod
    c1 = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            x = (i + 0.5) / N - 0.5
            y = (j + 0.5) / N - 0.5
            if x*x + y*y < 0.2**2:
                c1[i][j] = 1
    inits.append(c1)
    
    # 2. Ring
    c2 = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            x = (i + 0.5) / N - 0.5
            y = (j + 0.5) / N - 0.5
            r = math.sqrt(x*x + y*y)
            if 0.15 < r < 0.25:
                c2[i][j] = 1
    inits.append(c2)
    
    # 3. Random
    c3 = [[np.random.randint(2) for _ in range(N)] for _ in range(N)]
    # Enforce symmetry
    for i in range(N):
        for j in range(N):
            for (oi, oj) in get_orbit(i, j, N):
                c3[oi][oj] = c3[i][j]
    inits.append(c3)
    
    candidates = []
    
    for init in inits:
        current_cell = [row[:] for row in init]
        current_score = score_coarse(current_cell)
        best_local_cell = [row[:] for row in current_cell]
        best_local_score = current_score
        
        T = 0.5
        steps = 0
        while time.time() - start_time < time_limit - 20: # Leave 20s for full eval
            if steps > 500: break # Max steps per restart to allow multiple restarts?
            # Actually, let's just run until time limit or convergence
            if time.time() - start_time > time_limit - 20: break
            
            idx = np.random.randint(n_free)
            i, j = free_pixels[idx]
            orbit = get_orbit(i, j, N)
            
            # Flip
            for (oi, oj) in orbit:
                current_cell[oi][oj] = 1 - current_cell[oi][oj]
                
            new_score = score_coarse(current_cell)
            
            if new_score > current_score or np.random.rand() < math.exp((new_score - current_score) / T):
                current_score = new_score
                if new_score > best_local_score:
                    best_local_score = new_score
                    best_local_cell = [row[:] for row in current_cell]
            else:
                for (oi, oj) in orbit:
                    current_cell[oi][oj] = 1 - current_cell[oi][oj]
            
            T *= 0.995
            steps += 1
            
        candidates.append((best_local_score, best_local_cell))
        
        if time.time() - start_time > time_limit - 20:
            break
            
    # Evaluate top candidates with full solver
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    for score_c, cell in candidates[:5]:
        if time.time() - start_time > time_limit - 5:
            break
        score_f = score_full(cell)
        if score_f > best_score_full:
            best_score_full = score_f
            best_cell = cell
            
    if best_cell is None:
        best_cell = inits[0]
        
    return best_cell
