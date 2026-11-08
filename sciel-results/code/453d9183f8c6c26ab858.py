import numpy as np
import time
import random
import sys
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    TIME_LIMIT = 110.0  # Leave 10s safety margin for 120s limit
    
    n, m, p = 3, 3, 4
    dim_u = n * m  # 9
    dim_v = m * p  # 12
    dim_w = p * n  # 12
    
    # Construct target tensor T[i, j, k] = 1 if A[i,k]*B[k,j] contributes to C[i,j]
    # A is n x m (3x3), B is m x p (3x4)
    # C is n x p (3x4)
    # Tensor index convention:
    # U index: i * m + k (row i of A, col k) -> 0..8
    # V index: k * p + j (row k of B, col j) -> 0..11
    # W index: j * n + i (col j of C, row i) -> 0..11
    # Wait, the prompt says W[k*n+i, r]. Let's check the seed.
    # Seed: W[k*3+i, col] = 1.
    # Standard MM: C[i,j] = sum_k A[i,k] B[k,j].
    # Tensor T_{u,v,w} where u=(i,k), v=(k,j), w=(j,i).
    # T[(i*m+k), (k*p+j), (j*n+i)] = 1.
    
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def reconstruct(U, V, W, rank):
        # U: (dim_u, rank), V: (dim_v, rank), W: (dim_w, rank)
        # Return tensor
        T_rec = np.zeros((dim_u, dim_v, dim_w), dtype=float)
        for r in range(rank):
            T_rec += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        return T_rec

    def check_solution(U, V, W, rank):
        # Check if U, V, W are half-integers (or integers)
        # Multiply by 2 and check if close to integer
        U2 = U * 2
        V2 = V * 2
        W2 = W * 2
        
        if not np.allclose(U2, np.round(U2), atol=1e-5): return False
        if not np.allclose(V2, np.round(V2), atol=1e-5): return False
        if not np.allclose(W2, np.round(W2), atol=1e-5): return False
        
        # Check exact match with T
        U_int = np.round(U2).astype(int) / 2.0
        V_int = np.round(V2).astype(int) / 2.0
        W_int = np.round(W2).astype(int) / 2.0
        
        T_rec = reconstruct(U_int, V_int, W_int, rank)
        return np.allclose(T_rec, T, atol=1e-9)

    def try_find_rank(rank, time_budget):
        # Random restarts
        # We want to minimize || T - sum(U_r V_r W_r) ||^2
        # Flatten U, V, W into a single vector x
        # x = [U.flatten(), V.flatten(), W.flatten()]
        
        def objective(x):
            U = x[:dim_u * rank].reshape(dim_u, rank)
            V = x[dim_u * rank : dim_u * rank + dim_v * rank].reshape(dim_v, rank)
            W = x[dim_u * rank + dim_v * rank :].reshape(dim_w, rank)
            T_rec = reconstruct(U, V, W, rank)
            diff = T - T_rec
            return np.sum(diff * diff)
        
        best_loss = float('inf')
        best_x = None
        
        restarts = 0
        while time.time() - start_time < time_budget:
            restarts += 1
            # Initialize with small random values
            # Bias towards integer values? 
            # Strategy: Random small integers or half-integers
            scale = 1.0
            x0 = np.random.uniform(-scale, scale, size=(dim_u + dim_v + dim_w) * rank)
            
            # Try to use L-BFGS-B
            try:
                res = minimize(objective, x0, method='L-BFGS-B', 
                               options={'maxiter': 500, 'disp': False, 'maxls': 50})
                if res.fun < best_loss:
                    best_loss = res.fun
                    best_x = res.x
                    if best_loss < 1e-8:
                        # Found a perfect real-valued solution?
                        # Try rounding
                        U = best_x[:dim_u * rank].reshape(dim_u, rank)
                        V = best_x[dim_u * rank : dim_u * rank + dim_v * rank].reshape(dim_v, rank)
                        W = best_x[dim_u * rank + dim_v * rank :].reshape(dim_w, rank)
                        
                        # Round to nearest half-integer
                        U_round = np.round(U * 2) / 2.0
                        V_round = np.round(V * 2) / 2.0
                        W_round = np.round(W * 2) / 2.0
                        
                        if check_solution(U_round, V_round, W_round, rank):
                            return U_round, V_round, W_round
            except Exception:
                continue
                
        if best_x is not None and best_loss < 1e-4:
             # Try rounding even if not perfectly minimized
             U = best_x[:dim_u * rank].reshape(dim_u, rank)
             V = best_x[dim_u * rank : dim_u * rank + dim_v * rank].reshape(dim_v, rank)
             W = best_x[dim_u * rank + dim_v * rank :].reshape(dim_w, rank)
             U_round = np.round(U * 2) / 2.0
             V_round = np.round(V * 2) / 2.0
             W_round = np.round(W * 2) / 2.0
             if check_solution(U_round, V_round, W_round, rank):
                 return U_round, V_round, W_round
                 
        return None

    # Try ranks from 33 down to 30
    # Allocate time based on remaining time
    ranks_to_try = [33, 32, 31, 30]
    
    best_solution = None
    best_rank = 999
    
    # First, ensure we have the known rank 34 solution as fallback
    # Reconstructing rank 34 from scratch is safer than hardcoding if we want to be distinct, 
    # but the prompt says "rewrite... stronger algorithm". 
    # The hardcoded solution is part of the "current program". 
    # I will generate the rank 34 solution programmatically to be safe and distinct?
    # No, the prompt implies improving upon the current. 
    # If I fail to find 33, I should return 34.
    # I will hardcode the 34 solution to guarantee a valid baseline, 
    # but the code structure will be different (search loop).
    
    # Hardcoded Rank 34 from prompt
    U_34 = np.zeros((9, 34), dtype=float)
    V_34 = np.zeros((12, 34), dtype=float)
    W_34 = np.zeros((12, 34), dtype=float)
    
    col = 0
    # Strassen Left (A11 * B1L)
    U[0, col] = 1; U[4, col] = 1; V[0, col] = 1; V[5, col] = 1; W[0, col] = 1; W[4, col] = 1; col += 1
    U[3, col] = 1; U[4, col] = 1; V[0, col] = 1; W[1, col] = 1; W[4, col] = -1; col += 1
    U[0, col] = 1; V[1, col] = 1; V[5, col] = -1; W[3, col] = 1; W[4, col] = 1; col += 1
    U[4, col] = 1; V[4, col] = 1; V[0, col] = -1; W[0, col] = 1; W[1, col] = 1; col += 1
    U[0, col] = 1; U[1, col] = 1; V[5, col] = 1; W[0, col] = -1; W[3, col] = 1; col += 1
    U[3, col] = 1; U[0, col] = -1; V[0, col] = 1; V[1, col] = 1; W[4, col] = 1; col += 1
    U[1, col] = 1; U[4, col] = -1; V[4, col] = 1; V[5, col] = 1; W[0, col] = 1; col += 1
    
    # A12 * B2L
    U[2, col] = 1; V[8, col] = 1; W[0, col] = 1; col += 1
    U[2, col] = 1; V[9, col] = 1; W[3, col] = 1; col += 1
    U[5, col] = 1; V[8, col] = 1; W[1, col] = 1; col += 1
    U[5, col] = 1; V[9, col] = 1; W[4, col] = 1; col += 1
    
    # A21 * B1L
    U[6, col] = 1; V[0, col] = 1; W[2, col] = 1; col += 1
    U[6, col] = 1; V[1, col] = 1; W[5, col] = 1; col += 1
    U[7, col] = 1; V[4, col] = 1; W[2, col] = 1; col += 1
    U[7, col] = 1; V[5, col] = 1; W[5, col] = 1; col += 1
    
    # A22 * B2L
    U[8, col] = 1; V[8, col] = 1; W[2, col] = 1; col += 1
    U[8, col] = 1; V[9, col] = 1; W[5, col] = 1; col += 1
    
    # Strassen Right (A11 * B1R)
    U[0, col] = 1; U[4, col] = 1; V[2, col] = 1; V[7, col] = 1; W[6, col] = 1; W[10, col] = 1; col += 1
    U[3, col] = 1; U[4, col] = 1; V[2, col] = 1; W[7, col] = 1; W[10, col] = -1; col += 1
    U[0, col] = 1; V[3, col] = 1; V[7, col] = -1; W[9, col] = 1; W[10, col] = 1; col += 1
    U[4, col] = 1; V[6, col] = 1; V[2, col] = -1; W[6, col] = 1; W[7, col] = 1; col += 1
    U[0, col] = 1; U[1, col] = 1; V[7, col] = 1; W[6, col] = -1; W[9, col] = 1; col += 1
    U[3, col] = 1; U[0, col] = -1; V[2, col] = 1; V[3, col] = 1; W[10, col] = 1; col += 1
    U[1, col] = 1; U[4, col] = -1; V[6, col] = 1; V[7, col] = 1; W[6, col] = 1; col += 1
    
    # A12 * B2R
    U[2, col] = 1; V[10, col] = 1; W[6, col] = 1; col += 1
    U[2, col] = 1; V[11, col] = 1; W[9, col] = 1; col += 1
    U[5, col] = 1; V[10, col] = 1; W[7, col] = 1; col += 1
    U[5, col] = 1; V[11, col] = 1; W[10, col] = 1; col += 1
    
    # A21 * B1R
    U[6, col] = 1; V[2, col] = 1; W[8, col] = 1; col += 1
    U[6, col] = 1; V[3, col] = 1; W[11, col] = 1; col += 1
    U[7, col] = 1; V[6, col] = 1; W[8, col] = 1; col += 1
    U[7, col] = 1; V[7, col] = 1; W[11, col] = 1; col += 1
    
    # A22 * B2R
    U[8, col] = 1; V[10, col] = 1; W[8, col] = 1; col += 1
    U[8, col] = 1; V[11, col] = 1; W[11, col] = 1; col += 1
    
    # Verify hardcoded solution
    if check_solution(U, V, W, 34):
        best_solution = (U, V, W, 34)
        best_rank = 34
    else:
        # Fallback to trivial rank 36 if hardcoded fails (should not happen)
        rank = 36
        U_f = np.zeros((9, rank), dtype=float)
        V_f = np.zeros((12, rank), dtype=float)
        W_f = np.zeros((12, rank), dtype=float)
        c = 0
        for i in range(3):
            for j in range(3):
                for k in range(4):
                    U_f[i*3+k, c] = 1
                    V_f[k*4+j, c] = 1
                    W_f[j*3+i, c] = 1
                    c += 1
        best_solution = (U_f, V_f, W_f, 36)
        best_rank = 36

    # Now try to improve
    time_elapsed = time.time() - start_time
    time_remaining = TIME_LIMIT - time_elapsed
    
    if time_remaining > 5:
        # Split time among ranks
        # Give more time to higher probabilities? 
        # Just sequential.
        for target_rank in [33, 32]:
            if time.time() - start_time > TIME_LIMIT - 2:
                break
            
            time_budget = min(20.0, TIME_LIMIT - (time.time() - start_time) - 2.0)
            if time_budget < 1.0:
                break
                
            res = try_find_rank(target_rank, time_budget)
            if res is not None:
                U_new, V_new, W_new = res
                # Convert to lists for return
                if target_rank < best_rank:
                    best_solution = (U_new, V_new, W_new, target_rank)
                    best_rank = target_rank
                    # If we found 33, we can stop or try 32 briefly
                    # If we found 32, stop

    U_final, V_final, W_final, rank_final = best_solution
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

