import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 45 # Schoolbook
    best_U, best_V, best_W = None, None, None
    best_score = float('inf')
    
    # Precompute target tensor indices
    # T[i*m+j, j*p+k, k*n+i] = 1
    # We want sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    
    # We will search for ranks from 25 to 44
    # If we find a valid decomposition, we return it.
    
    start_time = time.time()
    time_limit = 110 # Safety margin
    
    for r in range(25, 45):
        if time.time() - start_time > time_limit:
            break
            
        # Try to find a solution for this rank
        found = False
        restarts = 500
        for _ in range(restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize U, V, W with random integers -1, 0, 1
            U = np.random.randint(-1, 2, (n*m, r))
            V = np.random.randint(-1, 2, (m*p, r))
            W = np.random.randint(-1, 2, (p*n, r))
            
            # Local search
            for step in range(100):
                if time.time() - start_time > time_limit:
                    break
                    
                # Calculate error
                # Error = sum over all i,j,k of (sum_r U[i,j,r]V[j,k,r]W[k,i,r] - 1)^2
                # This is expensive to compute fully each step.
                # We can compute the product tensor P and compare with T.
                
                # Efficiently compute P
                # P[i*m+j, j*p+k, k*n+i]
                # This is a 3D tensor.
                # We can flatten it.
                
                # Let's just compute the error for a random subset of indices to speed up?
                # No, we need exact 0.
                
                # Compute full error
                # U shape (9, r), V shape (15, r), W shape (15, r)
                # P is 9x15x15? No.
                # Indices: i (0..2), j (0..2), k (0..4)
                # U index: i*m+j
                # V index: j*p+k
                # W index: k*n+i
                
                # We can use einsum or loops.
                # For speed, we can precompute the indices.
                
                # Let's just do a simple check:
                # If error is 0, we found it.
                
                # To speed up, we can check a random subset of 100 indices.
                # If subset error is 0, check full.
                
                idx_i = np.random.randint(0, n, 100)
                idx_j = np.random.randint(0, m, 100)
                idx_k = np.random.randint(0, p, 100)
                
                u_idx = idx_i * m + idx_j
                v_idx = idx_j * p + idx_k
                w_idx = idx_k * n + idx_i
                
                # Sum over r
                # U[u_idx, :] * V[v_idx, :] * W[w_idx, :]
                # This is element-wise product of 3 vectors of length r, then sum.
                # We can do this with matrix multiplication?
                # Let A = U[u_idx, :], B = V[v_idx, :], C = W[w_idx, :]
                # Error term = sum_r A[:, r] * B[:, r] * C[:, r]
                # This is not a standard dot product.
                
                # We can compute it as:
                # S = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :], axis=1)
                # Error = np.sum((S - 1)**2)
                
                S = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :], axis=1)
                err_subset = np.sum((S - 1)**2)
                
                if err_subset == 0:
                    # Check full
                    full_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                ui = i * m + j
                                vi = j * p + k
                                wi = k * n + i
                                S_full = np.sum(U[ui, :] * V[vi, :] * W[wi, :], axis=1)
                                full_err += (S_full - 1)**2
                    if full_err == 0:
                        # Found!
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
                # Flip a random entry
                mat = random.choice([U, V, W])
                row = random.randint(0, mat.shape[0]-1)
                col = random.randint(0, r-1)
                old_val = mat[row, col]
                mat[row, col] = random.randint(-1, 2)
                
                # Recalculate subset error to decide acceptance
                # This is slow.
                # Simple hill climbing: always accept? Or accept if better?
                # Let's just accept if the new subset error is better.
                # But we changed one entry, so only indices involving that entry change.
                # This is complex.
                # Let's just do a simple random walk and check every 10 steps.
                
    # If no better solution found, return schoolbook
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
