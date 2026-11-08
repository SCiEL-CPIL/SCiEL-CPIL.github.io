import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # Target: 2x3 * 3x9
    # We try to find a rank r_small for 2x3 * 3x3
    # and then tile it 3 times to get rank 3*r_small for 2x3 * 3x9.
    
    # Schoolbook rank for 2x3x9 is 54.
    # Schoolbook rank for 2x3x3 is 18.
    # We search for r_small < 18.
    
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s margin
    
    # Target tensor for 2x3x3
    n_s, m_s, p_s = 2, 3, 3
    target_tensor = np.zeros((n_s * m_s, m_s * p_s, p_s * n_s))
    for i in range(n_s):
        for j in range(m_s):
            for k in range(p_s):
                target_tensor[i * m_s + j, j * p_s + k, k * n_s + i] = 1.0
    
    best_rank = 18
    best_U_small = None
    best_V_small = None
    best_W_small = None
    
    # Search for ranks 12 to 17
    for r in range(12, 18):
        if time.time() > deadline:
            break
            
        # Try multiple random restarts
        found = False
        max_restarts = 50
        restarts_done = 0
        
        while restarts_done < max_restarts and not found:
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n_s * m_s, r)
            V = np.random.randn(m_s * p_s, r)
            W = np.random.randn(p_s * n_s, r)
            
            # Use scipy optimize
            try:
                from scipy.optimize import minimize
                def objective(params):
                    U_p = params[:n_s*m_s*r].reshape(n_s*m_s, r)
                    V_p = params[n_s*m_s*r:n_s*m_s*r + m_s*p_s*r].reshape(m_s*p_s, r)
                    W_p = params[n_s*m_s*r + m_s*p_s*r:].reshape(p_s*n_s, r)
                    # Reconstruct tensor
                    # T = sum_t U[:,t] outer V[:,t] outer W[:,t]
                    # This is expensive to do in a loop, but r is small.
                    # We can use einsum or manual loops.
                    # T[i, j, k] = sum_t U[i,t] V[j,t] W[k,t]
                    # We want to minimize || T_reconstructed - target ||_F
                    # T_reconstructed is (n_s*m_s, m_s*p_s, p_s*n_s)
                    # We can compute it as:
                    # T_recon = np.einsum('it,jt,kt->ijk', U_p, V_p, W_p)
                    # But we need to match indices.
                    # U index: i*m_s+j
                    # V index: j*p_s+k
                    # W index: k*n_s+i
                    # This mapping is not a simple outer product of flattened vectors in the standard sense
                    # unless we permute the tensor.
                    
                    # Let's permute the target tensor to standard form.
                    # Standard form: T_std[a, b, c] where a indexes U, b indexes V, c indexes W.
                    # Our indices:
                    # a = i*m_s + j
                    # b = j*p_s + k
                    # c = k*n_s + i
                    # This is a specific permutation of the standard tensor.
                    # Instead of permuting, we can just check the error on the original indices.
                    
                    # Error calculation:
                    # E = 0
                    # for t in range(r):
                    #   for i,j in U_p[:,t] non-zero? No, U_p is dense.
                    #   We can just compute the full tensor and compare.
                    
                    # Efficient way:
                    # Compute T_recon in the permuted space?
                    # Let's just compute the full tensor.
                    # Size is 6x9x6 = 324 elements.
                    # r is up to 17.
                    # 17 iterations of outer products?
                    
                    # T_recon = np.zeros((n_s*m_s, m_s*p_s, p_s*n_s))
                    # for t in range(r):
                    #    T_recon += np.einsum('i,j,k->ijk', U_p[:,t], V_p[:,t], W_p[:,t])
                    # This is O(r * 6 * 9 * 6) = 17 * 324 ~ 5500 ops. Very fast.
                    
                    T_recon = np.zeros((n_s*m_s, m_s*p_s, p_s*n_s))
                    for t in range(r):
                        T_recon += np.einsum('i,j,k->ijk', U_p[:,t], V_p[:,t], W_p[:,t])
                    
                    diff = T_recon - target_tensor
                    return np.sum(diff * diff)
                
                # Initial params
                p0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                
                res = minimize(objective, p0, method='BFGS', options={'maxiter': 100, 'gtol': 1e-6})
                
                if res.success or res.fun < 1e-5:
                    # Check for exactness by rounding
                    U_r = np.round(res.x[:n_s*m_s*r].reshape(n_s*m_s, r) * 2) / 2
                    V_r = np.round(res.x[n_s*m_s*r:n_s*m_s*r + m_s*p_s*r].reshape(m_s*p_s, r) * 2) / 2
                    W_r = np.round(res.x[n_s*m_s*r + m_s*p_s*r:].reshape(p_s*n_s, r) * 2) / 2
                    
                    T_check = np.zeros((n_s*m_s, m_s*p_s, p_s*n_s))
                    for t in range(r):
                        T_check += np.einsum('i,j,k->ijk', U_r[:,t], V_r[:,t], W_r[:,t])
                    
                    if np.allclose(T_check, target_tensor, atol=1e-9):
                        best_rank = r
                        best_U_small = U_r
                        best_V_small = V_r
                        best_W_small = W_r
                        found = True
                        break
            except Exception as e:
                pass
            
            restarts_done += 1
        
        if found:
            break

    # Construct final 2x3x9 decomposition
    if best_rank < 18:
        r_total = 3 * best_rank
        U_final = np.zeros((n * m, r_total))
        V_final = np.zeros((m * p, r_total))
        W_final = np.zeros((p * n, r_total))
        
        for block in range(3):
            U_final[:, block*best_rank:(block+1)*best_rank] = best_U_small
            # V and W need to be shifted
            # V indices: j*p + k. For block 'block', k is offset by block*3.
            # So V_final row (j*9 + block*3 + k_small) gets best_V_small row (j*3 + k_small)
            # Wait, best_V_small has rows indexed by j*3 + k_small (0..8).
            # In the final V, the row index is j*9 + k_final.
            # k_final = block*3 + k_small.
            # So V_final[j*9 + block*3 + k_small, col] = best_V_small[j*3 + k_small, col]
            
            for j in range(m_s):
                for k_s in range(p_s):
                    row_final = j * p + block * p_s + k_s
                    row_small = j * m_s + k_s # Wait, m_s=3, p_s=3.
                    # best_V_small row index is j*p_s + k_s = 3j + k_s.
                    V_final[row_final, block*best_rank:(block+1)*best_rank] = best_V_small[3*j + k_s, :]
                    
            # W indices: k*n + i. For block 'block', k is offset by block*3.
            # W_final row (block*3 + k_small)*n + i gets best_W_small row k_small*n + i
            for k_s in range(p_s):
                for i in range(n_s):
                    row_final = (block * p_s + k_s) * n + i
                    row_small = k_s * n_s + i
                    W_final[row_final, block*best_rank:(block+1)*best_rank] = best_W_small[k_s * n_s + i, :]
        
        # Convert to lists of lists for output
        U_list = U_final.tolist()
        V_list = V_final.tolist()
        W_list = W_final.tolist()
        
        # Ensure integer/half-integer
        # They are already rounded to 0.5
        
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fall back to schoolbook
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
