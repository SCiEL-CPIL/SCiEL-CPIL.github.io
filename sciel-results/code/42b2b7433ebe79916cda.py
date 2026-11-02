"""Matrix multiplication tensor decomposition search for 2x3 * 3x4."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 4]
DEADLINE_MARGIN = 10.0  # seconds before the 150s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    # Dimensions:
    # U: (n*m, r) -> (6, r)
    # V: (m*p, r) -> (12, r)
    # W: (p*n, r) -> (8, r)
    
    # Target tensor T[i, j, k] = 1 for all i in [0,2), j in [0,3), k in [0,4)
    # Decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Helper to check if a decomposition is valid (exact within tolerance)
    def is_valid(U, V, W, tol=1e-9):
        r = U.shape[1]
        # Reshape to (n, m, r), (m, p, r), (p, n, r)
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        
        # Compute S[i, j, k] = sum_c U_r[i, j, c] * V_r[j, k, c] * W_r[k, i, c]
        # Note: W index is k*n+i. In W_r (p, n, r), index is [k, i, c].
        # So we need W_r[k, i, c].
        
        S = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        err = np.max(np.abs(S - 1.0))
        return err < tol, err

    # Helper to snap to half-integers
    def snap_half(x):
        return np.round(x * 2.0) / 2.0

    # Schoolbook decomposition (Rank 24)
    rank_school = n * m * p
    U0 = np.zeros((n*m, rank_school))
    V0 = np.zeros((m*p, rank_school))
    W0 = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank_school
    
    # Search for lower ranks
    # Known minimal rank for 2x3x4 is 17.
    # We search from 14 to 23.
    for r in range(14, rank_school):
        if time.time() - start_time > 140:
            break
            
        found = False
        # Try a few random initializations for this rank
        for attempt in range(20):
            if time.time() - start_time > 140:
                break
                
            # Random initialization
            # Scale matters. 1/sqrt(r) or similar.
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            # Gradient Descent
            # We want to minimize sum (S_ijk - 1)^2
            # S_ijk = sum_c U_ij_c V_jk_c W_ki_c
            
            # Pre-allocate gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            lr = 0.05
            # Adaptive learning rate
            prev_err = float('inf')
            
            for step in range(1000):
                if time.time() - start_time > 140:
                    break
                    
                U_r = U.reshape(n, m, r)
                V_r = V.reshape(m, p, r)
                W_r = W.reshape(p, n, r)
                
                # S[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
                S = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                
                err = np.sum((S - 1.0)**2)
                
                if err < 1e-10:
                    break
                
                # If error increased significantly, reduce learning rate
                if err > prev_err * 1.5:
                    lr *= 0.5
                else:
                    # Slowly increase if stable? Or keep constant.
                    pass
                prev_err = err
                
                # Gradients
                # dU[i, j, c] = sum_k 2(S[i, j, k] - 1) * V[j, k, c] * W[k, i, c]
                # dV[j, k, c] = sum_i 2(S[i, j, k] - 1) * U[i, j, c] * W[k, i, c]
                # dW[k, i, c] = sum_j 2(S[i, j, k] - 1) * U[i, j, c] * V[j, k, c]
                
                # Using einsum for gradients
                # Note: S is (n, m, p). U is (n, m, r). V is (m, p, r). W is (p, n, r).
                
                # dU: sum over k
                # (S-1) is (n, m, p)
                # V is (m, p, r)
                # W is (p, n, r)
                # We need sum_k (S[i,j,k]-1) * V[j,k,c] * W[k,i,c]
                # This is tricky with direct einsum because of the index permutations.
                
                # Let's compute intermediates.
                # T1[i, j, c] = sum_k (S[i,j,k]-1) * V[j,k,c] * W[k,i,c]
                # This is not a standard tensor product.
                
                # Alternative:
                # dU = 2 * ( (S-1) * (V * W) ) contracted over k?
                
                # Let's do it manually with loops for clarity and correctness, 
                # as the dimensions are small (2x3x4).
                # This is fast enough.
                
                dU.fill(0)
                dV.fill(0)
                dW.fill(0)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            e = 2.0 * (S[i, j, k] - 1.0)
                            for c in range(r):
                                dU[i*m+j, c] += e * V[j*p+k, c] * W[k*n+i, c]
                                dV[j*p+k, c] += e * U[i*m+j, c] * W[k*n+i, c]
                                dW[k*n+i, c] += e * U[i*m+j, c] * V[j*p+k, c]
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
            # Check convergence
            valid, err = is_valid(U, V, W)
            if valid:
                # Snap to half-integers
                U_s = snap_half(U)
                V_s = snap_half(V)
                W_s = snap_half(W)
                
                valid_s, err_s = is_valid(U_s, V_s, W_s)
                if valid_s:
                    best_U, best_V, best_W = U_s, V_s, W_s
                    best_rank = r
                    found = True
                    break
                else:
                    # Maybe the float solution is good but snapping broke it?
                    # The problem requires integer/half-integer entries.
                    # If snapping fails, this rank is not valid under the constraint.
                    # We should only accept if snapped version is valid.
                    pass
        
        if found:
            # We found a valid decomposition of rank r.
            # Since we iterate r upwards, this is the minimal rank found so far.
            # We can break if we are confident this is the global minimum.
            # Given the search space, finding 17 is likely optimal.
            # We can break to save time, or continue to see if 16 is possible.
            # Let's break to ensure we return within time limit.
            break

    # If no better rank found, return schoolbook (which is valid)
    # The schoolbook matrices are already integers (0 or 1), so they are valid half-integers.
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
