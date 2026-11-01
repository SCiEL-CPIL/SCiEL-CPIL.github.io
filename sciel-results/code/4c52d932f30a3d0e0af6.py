import time
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Target rank is 14
    target_rank = 14
    
    # Dimensions of U, V, W
    dim_u = n * m  # 6
    dim_v = m * p  # 12
    dim_w = p * n  # 8
    
    # Precompute indices for fast evaluation
    # We need to compute sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Let's flatten the indices for the tensor T
    # T has shape (n, m, p)
    
    def compute_error(U, V, W):
        """
        Compute the sum of squared errors.
        U: (n*m, r)
        V: (m*p, r)
        W: (p*n, r)
        """
        r = U.shape[1]
        U_mat = U.reshape(n, m, r)
        V_mat = V.reshape(m, p, r)
        W_mat = W.reshape(p, n, r)
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U_mat[i, j, c] * V_mat[j, k, c] * W_mat[k, i, c]
                    error += (1.0 - val) ** 2
        return error

    def is_valid(U, V, W):
        return compute_error(U, V, W) < 1e-9

    # Search for a solution
    # We try ranks from 14 up to 24
    # But we know 14 is possible.
    
    # Strategy: Local search with random restarts
    # Initialize U, V, W with random integers in {-2, -1, 0, 1, 2}
    
    best_rank = 24
    best_solution = None
    
    # Fallback schoolbook solution
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_solution = (U_school, V_school, W_school)
    
    # Try to find rank 14
    if time.time() < deadline:
        # Use a random search with local improvement
        # We will try a few hundred restarts
        
        np.random.seed(42) # For reproducibility in testing, but maybe random is better?
        
        for attempt in range(500):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randint(-2, 3, size=(dim_u, target_rank)).astype(float)
            V = np.random.randint(-2, 3, size=(dim_v, target_rank)).astype(float)
            W = np.random.randint(-2, 3, size=(dim_w, target_rank)).astype(float)
            
            current_error = compute_error(U, V, W)
            
            # Local search: try to change one element at a time
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Randomly pick a matrix and an element
                mat_choice = np.random.randint(0, 3)
                if mat_choice == 0:
                    row, col_idx = np.random.randint(0, dim_u), np.random.randint(0, target_rank)
                    old_val = U[row, col_idx]
                    # Try small changes
                    for delta in [-1, -0.5, 0.5, 1]:
                        new_val = old_val + delta
                        if new_val == old_val: continue
                        U[row, col_idx] = new_val
                        new_error = compute_error(U, V, W)
                        if new_error < current_error:
                            current_error = new_error
                            improved = True
                            break
                        else:
                            U[row, col_idx] = old_val
                            
                elif mat_choice == 1:
                    row, col_idx = np.random.randint(0, dim_v), np.random.randint(0, target_rank)
                    old_val = V[row, col_idx]
                    for delta in [-1, -0.5, 0.5, 1]:
                        new_val = old_val + delta
                        if new_val == old_val: continue
                        V[row, col_idx] = new_val
                        new_error = compute_error(U, V, W)
                        if new_error < current_error:
                            current_error = new_error
                            improved = True
                            break
                        else:
                            V[row, col_idx] = old_val
                            
                else:
                    row, col_idx = np.random.randint(0, dim_w), np.random.randint(0, target_rank)
                    old_val = W[row, col_idx]
                    for delta in [-1, -0.5, 0.5, 1]:
                        new_val = old_val + delta
                        if new_val == old_val: continue
                        W[row, col_idx] = new_val
                        new_error = compute_error(U, V, W)
                        if new_error < current_error:
                            current_error = new_error
                            improved = True
                            break
                        else:
                            W[row, col_idx] = old_val
                
                if current_error < 1e-9:
                    # Found a solution!
                    # Check if it uses half-integers or integers
                    # The problem allows integer/half-integer
                    # Our deltas are 0.5, so we stay on the half-integer grid
                    best_rank = target_rank
                    best_solution = (U, V, W)
                    break
            
            if best_rank == target_rank:
                break
        
    U, V, W = best_solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
