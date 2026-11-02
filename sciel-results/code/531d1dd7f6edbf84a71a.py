import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    def try_rank(rank, max_trials=1000):
        if time.time() > deadline - 15:
            return None
        
        target = np.ones((n, m, p))
        target_flat = target.flatten()
        
        # Precompute index mappings for efficiency
        # For each equation (i,j,k), we need to know which columns of A_mat are non-zero
        # A_mat[eq_idx, col_idx] = U[i*m+j, r] * V[j*p+k, r]
        
        for trial in range(max_trials):
            if time.time() > deadline - 15:
                return None
            
            # Random integer U, V in {-1, 0, 1}
            U = np.random.randint(-1, 2, (n * m, rank))
            V = np.random.randint(-1, 2, (m * p, rank))
            
            # Build the linear system more efficiently using vectorized operations
            # A_mat has shape (n*m*p, p*n*rank)
            # A_mat[i*m*p + j*p + k, k*n*rank + i*rank + r] = U[i*m+j, r] * V[j*p+k, r]
            
            # Create indices
            i_idx = np.repeat(np.arange(n), m * p)
            j_idx = np.tile(np.repeat(np.arange(m), p), n)
            k_idx = np.tile(np.arange(p), n * m)
            
            # For each (i,j,k), the row index is i*m*p + j*p + k
            row_idx = i_idx * m * p + j_idx * p + k_idx
            
            # For each r, and for each (i,j,k), the column is k*n*rank + i*rank + r
            # We need to create the full A_mat
            
            # More efficient: build A_mat column by column
            A_mat = np.zeros((n * m * p, p * n * rank))
            
            for r in range(rank):
                # For column r of each (k,i) block
                # A_mat[row, k*n*rank + i*rank + r] = U[i*m+j, r] * V[j*p+k, r]
                # for all (i,j,k)
                
                for k in range(p):
                    for i in range(n):
                        col_base = k * n * rank + i * rank + r
                        # For all j:
                        # rows: i*m*p + j*p + k for j in range(m)
                        # values: U[i*m+j, r] * V[j*p+k, r]
                        for j in range(m):
                            row = i * m * p + j * p + k
                            A_mat[row, col_base] = U[i * m + j, r] * V[j * p + k, r]
            
            # Solve
            try:
                W_flat, residuals, rank_sv, sv = np.linalg.lstsq(A_mat, target_flat, rcond=None)
            except:
                continue
            
            # Check if solution is close to integer
            W_int = np.round(W_flat)
            error = np.linalg.norm(A_mat @ W_int - target_flat)
            
            if error < 1e-9:
                W = W_int.reshape(p * n, rank)
                return U, V, W
        
        return None
    
    # Try ranks from lower to higher
    # Known lower bounds suggest rank >= 30 for 3x4x4
    # Try from 35 upward
    for rank in range(35, 48):
        if time.time() > deadline - 20:
            break
        
        # Decrease trials as rank increases (harder to find)
        trials = 800 if rank < 40 else 500
        result = try_rank(rank, max_trials=trials)
        if result is not None:
            U, V, W = result
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    return schoolbook()
