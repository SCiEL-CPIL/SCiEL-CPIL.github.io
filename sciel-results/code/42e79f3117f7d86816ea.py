import numpy as np
import time
import random

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Schoolbook decomposition
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    def get_schoolbook():
        return {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }

    start_time = time.time()
    deadline = start_time + 140 # 10s margin

    # Target ranks to search for
    target_ranks = [15, 16, 17]
    
    # Values to sample from
    # Using half-integers
    vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    def compute_error(U, V, W):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        r = U.shape[1]
        # Reshape to (n, m, r), (m, p, r), (p, n, r)
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        
        # T_ijk = sum_r U_ijr V_jkr W_kin
        # Note: W index is k*n+i, so W_reshaped is (p, n, r) -> W[k, i, r]
        # The sum is over r.
        # T[i, j, k] = np.einsum('ijr,jkr,kin->ijk', U_reshaped, V_reshaped, W_reshaped)
        T = np.einsum('ijr,jkr,kin->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        # Target is 1 for all i, j, k
        target = np.ones((n, m, p))
        err = np.sum((T - target) ** 2)
        return err

    def local_search(r, max_iter=50000):
        # Initialize
        U = np.random.choice(vals, size=(n * m, r))
        V = np.random.choice(vals, size=(m * p, r))
        W = np.random.choice(vals, size=(p * n, r))
        
        current_err = compute_error(U, V, W)
        
        # If we start with a good guess, we might be close
        # But random is usually far.
        
        for it in range(max_iter):
            if time.time() > deadline:
                return None
            
            if current_err < 1e-9:
                return U, V, W
            
            # Pick a random matrix
            mat_idx = random.randint(0, 2)
            if mat_idx == 0:
                mat = U
                rows = n * m
            elif mat_idx == 1:
                mat = V
                rows = m * p
            else:
                mat = W
                rows = p * n
            
            row = random.randint(0, rows - 1)
            col = random.randint(0, r - 1)
            
            old_val = mat[row, col]
            
            # Try neighbors
            # Neighbors of 0: -0.5, 0.5
            # Neighbors of 0.5: 0, 1
            # Neighbors of 1: 0.5
            # etc.
            neighbors = []
            if old_val == -1.0:
                neighbors = [-0.5, 0.0]
            elif old_val == -0.5:
                neighbors = [-1.0, 0.0, 0.5]
            elif old_val == 0.0:
                neighbors = [-0.5, 0.5]
            elif old_val == 0.5:
                neighbors = [0.0, 1.0, -0.5]
            elif old_val == 1.0:
                neighbors = [0.5, 0.0]
            
            if not neighbors:
                continue
                
            new_val = random.choice(neighbors)
            
            mat[row, col] = new_val
            new_err = compute_error(U, V, W)
            
            if new_err < current_err:
                current_err = new_err
            else:
                mat[row, col] = old_val
                
        return None

    best_result = None
    best_rank = rank_schoolbook
    
    # Try to find a lower rank
    for r in target_ranks:
        if time.time() > deadline:
            break
            
        # Multiple restarts
        restarts = 50
        for _ in range(restarts):
            if time.time() > deadline:
                break
                
            # Run local search with limited iterations to allow more restarts
            # 5000 iterations per restart * 50 restarts = 250k iterations
            # compute_error is fast with einsum
            res = local_search(r, max_iter=2000)
            if res:
                U, V, W = res
                # Verify exactly
                if compute_error(U, V, W) < 1e-9:
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    best_rank = r
                    break
        
        if best_result:
            break

    if best_result:
        return best_result
    else:
        return get_schoolbook()

