"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # 1. Construct Schoolbook Solution (Rank = n*m*p)
    U0 = np.zeros((n * m, rank_school))
    V0 = np.zeros((m * p, rank_school))
    W0 = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank_school
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 2. Try to find lower rank solutions using Warm-Start ALS
    # We try ranks from rank_school - 1 down to a lower bound.
    # Lower bound is max(n*m, m*p, p*n) = max(15, 25, 15) = 25.
    # We don't need to check all ranks, just a few candidates.
    candidate_ranks = [74, 70, 65, 60, 50]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
            
        # Initialize from Schoolbook: Take first target_rank columns
        U_curr = U0[:, :target_rank].copy()
        V_curr = V0[:, :target_rank].copy()
        W_curr = W0[:, :target_rank].copy()
        
        # Run ALS iterations
        max_iterations = 20
        converged = False
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            # --- Update U ---
            # For each (i, j), solve (sum_k V[j,k,:] * W[k,i,:].T) @ U[i,j,:] = 1
            # V shape: (m*p, r) -> V[j*p + k, :] is vector of size r
            # W shape: (p*n, r) -> W[k*n + i, :] is vector of size r
            # M_ij = sum_k outer(V[j*p+k, :], W[k*n+i, :])
            
            U_new = np.zeros_like(U_curr)
            # Batch construction of M
            # M shape: (n*m, r, r)
            M_batch = np.zeros((n * m, target_rank, target_rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        v_vec = V_curr[j * p + k, :]
                        w_vec = W_curr[k * n + i, :]
                        M_batch[i * m + j] += np.outer(v_vec, w_vec)
            
            # Solve M_batch @ U_new.T = 1
            # U_new.T shape: (n*m, r, 1)
            try:
                # np.linalg.solve supports (..., M, M) and (..., M, K)
                U_new_T = np.linalg.solve(M_batch + 1e-8 * np.eye(target_rank), np.ones((n * m, target_rank, 1)))
                U_new = U_new_T.squeeze(-1).T
            except:
                # Fallback to lstsq if singular
                U_new_T, _, _, _ = np.linalg.lstsq(M_batch + 1e-8 * np.eye(target_rank), np.ones((n * m, target_rank, 1)), rcond=None)
                U_new = U_new_T.squeeze(-1).T
            
            U_curr = U_new
            
            # --- Update V ---
            # For each (j, k), solve (sum_i U[i,j,:] * W[k,i,:].T) @ V[j,k,:] = 1
            V_new = np.zeros_like(V_curr)
            M_batch_v = np.zeros((m * p, target_rank, target_rank))
            for j in range(m):
                for k in range(p):
                    for i in range(n):
                        u_vec = U_curr[i * m + j, :]
                        w_vec = W_curr[k * n + i, :]
                        M_batch_v[j * p + k] += np.outer(u_vec, w_vec)
            
            try:
                V_new_T = np.linalg.solve(M_batch_v + 1e-8 * np.eye(target_rank), np.ones((m * p, target_rank, 1)))
                V_new = V_new_T.squeeze(-1).T
            except:
                V_new_T, _, _, _ = np.linalg.lstsq(M_batch_v + 1e-8 * np.eye(target_rank), np.ones((m * p, target_rank, 1)), rcond=None)
                V_new = V_new_T.squeeze(-1).T
                
            V_curr = V_new
            
            # --- Update W ---
            # For each (k, i), solve (sum_j U[i,j,:] * V[j,k,:].T) @ W[k,i,:] = 1
            W_new = np.zeros_like(W_curr)
            M_batch_w = np.zeros((p * n, target_rank, target_rank))
            for k in range(p):
                for i in range(n):
                    for j in range(m):
                        u_vec = U_curr[i * m + j, :]
                        v_vec = V_curr[j * p + k, :]
                        M_batch_w[k * n + i] += np.outer(u_vec, v_vec)
            
            try:
                W_new_T = np.linalg.solve(M_batch_w + 1e-8 * np.eye(target_rank), np.ones((p * n, target_rank, 1)))
                W_new = W_new_T.squeeze(-1).T
            except:
                W_new_T, _, _, _ = np.linalg.lstsq(M_batch_w + 1e-8 * np.eye(target_rank), np.ones((p * n, target_rank, 1)), rcond=None)
                W_new = W_new_T.squeeze(-1).T
                
            W_curr = W_new
            
            # Check Error
            # Error = sum_{i,j,k} (sum_c U[i,j,c] V[j,k,c] W[k,i,c] - 1)^2
            # We can compute this efficiently using tensor contractions or just loops for small sizes
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_curr[i * m + j, :], V_curr[j * p + k, :])
                        val = np.dot(val, W_curr[k * n + i, :])
                        err += (val - 1.0)**2
                        
            if err < 1e-8:
                converged = True
                break
        
        if converged:
            best_rank = target_rank
            best_U, best_V, best_W = U_curr, V_curr, W_curr
            break # Found a valid lower rank, stop searching for lower ones immediately? 
                  # Or continue to find even lower? The loop goes down, so we can continue.
                  # But usually the first found is good enough for a jump. Let's continue to try lower.
    
    # Return best found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
