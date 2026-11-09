import numpy as np
import random
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Target tensor T
    target_tensor = np.zeros((dim_u, dim_v, dim_w), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_tensor[u, v, w] = 1
    target_flat = target_tensor.flatten()

    def compute_error(U, V, W):
        # Fast error calculation: sum of squared differences
        # T_recon = sum_c U[:,c] V[:,c] W[:,c]
        T_recon = np.zeros((dim_u, dim_v, dim_w), dtype=int)
        r = U.shape[1]
        for c in range(r):
            T_recon += U[:, c][:, None, None] * V[:, c][None, :, None] * W[:, c][None, None, :]
        diff = T_recon - target_tensor
        return np.sum(diff * diff)

    def random_matrix(rows, cols, density=0.5):
        mat = np.zeros((rows, cols), dtype=int)
        mask = np.random.rand(rows, cols) < density
        mat[mask] = np.random.choice([-1, 1], size=np.sum(mask))
        return mat

    start_time = time.time()
    time_limit = 110  # seconds, leave buffer

    # Search for minimal rank
    # Schoolbook rank is 24.
    # We try to find rank < 24.
    for r in range(18, 25):
        if time.time() - start_time > time_limit:
            break
        
        # Multiple restarts
        for restart in range(300):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize
            U = random_matrix(dim_u, r, 0.6)
            V = random_matrix(dim_v, r, 0.6)
            W = random_matrix(dim_w, r, 0.6)
            
            current_error = compute_error(U, V, W)
            best_error = current_error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Local Search: Hill Climbing with Random Restarts
            # Max steps
            max_steps = 200
            stuck_counter = 0
            step = 0
            
            while step < max_steps:
                if time.time() - start_time > time_limit:
                    break
                
                if best_error == 0:
                    return {
                        "U": best_U.tolist(),
                        "V": best_V.tolist(),
                        "W": best_W.tolist()
                    }
                
                # Pick a random element to mutate
                mat_choice = np.random.choice([0, 1, 2])
                if mat_choice == 0:
                    idx = (np.random.randint(dim_u), np.random.randint(r))
                    mat = U
                elif mat_choice == 1:
                    idx = (np.random.randint(dim_v), np.random.randint(r))
                    mat = V
                else:
                    idx = (np.random.randint(dim_w), np.random.randint(r))
                    mat = W
                    
                old_val = mat[idx]
                # Choose new value from {-1, 0, 1}
                # Avoid same value
                new_val = np.random.choice([-1, 0, 1])
                while new_val == old_val:
                    new_val = np.random.choice([-1, 0, 1])
                    
                mat[idx] = new_val
                new_error = compute_error(U, V, W)
                
                if new_error < current_error:
                    current_error = new_error
                    stuck_counter = 0
                    if new_error < best_error:
                        best_error = new_error
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                else:
                    # Revert
                    mat[idx] = old_val
                    stuck_counter += 1
                    
                    # If stuck, perturb
                    if stuck_counter > 20:
                        for _ in range(10):
                            mc = np.random.choice([0, 1, 2])
                            if mc == 0:
                                idx2 = (np.random.randint(dim_u), np.random.randint(r))
                                U[idx2] = np.random.choice([-1, 0, 1])
                            elif mc == 1:
                                idx2 = (np.random.randint(dim_v), np.random.randint(r))
                                V[idx2] = np.random.choice([-1, 0, 1])
                            else:
                                idx2 = (np.random.randint(dim_w), np.random.randint(r))
                                W[idx2] = np.random.choice([-1, 0, 1])
                        current_error = compute_error(U, V, W)
                        stuck_counter = 0
                step += 1

    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((dim_u, rank), dtype=int)
    V = np.zeros((dim_v, rank), dtype=int)
    W = np.zeros((dim_w, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U[u, col] = 1
                V[v, col] = 1
                W[w, col] = 1
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
