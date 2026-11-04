import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    candidates = np.array([-2, -1, 0, 1, 2])
    
    best_rank = nm * mp * pn
    best_decomp = None
    
    # Initialize with schoolbook
    U_init = np.zeros((nm, best_rank), dtype=int)
    V_init = np.zeros((mp, best_rank), dtype=int)
    W_init = np.zeros((pn, best_rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_init[i * m + j, col] = 1
                V_init[j * p + k, col] = 1
                W_init[k * n + i, col] = 1
                col += 1
    best_decomp = {"U": U_init.tolist(), "V": V_init.tolist(), "W": W_init.tolist()}
    
    # Precompute indices for error calculation
    # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # This is a 3D tensor contraction.
    
    # We will try ranks from 24 up to 35
    for r in range(24, best_rank):
        if time.time() > deadline:
            break
            
        found = False
        restarts = 0
        max_restarts = 100
        
        while restarts < max_restarts and not found:
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.choice(candidates, size=(nm, r))
            V = np.random.choice(candidates, size=(mp, r))
            W = np.random.choice(candidates, size=(pn, r))
            
            def get_error(U, V, W):
                # Compute the tensor product sum
                # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
                # Sum over r: U[i,j,c] * V[j,k,c] * W[k,i,c]
                # This is a specific contraction.
                # We can use np.einsum or manual loops if small.
                # n=3, m=3, p=4, r~24.
                # 3*3*4 = 36 terms.
                # For each term, sum over r.
                # 36 * r operations.
                # This is fast enough if vectorized.
                
                # Let's construct the full tensor T of size (n, m, p)
                # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                
                # We can iterate over i, j, k
                errs = np.zeros((n, m, p), dtype=int)
                for i in range(n):
                    for j in range(m):
                        u_row = U[i * m + j]
                        for k in range(p):
                            v_row = V[j * p + k]
                            w_row = W[k * n + i]
                            # Dot product of u_row, v_row, w_row? No, element-wise product sum
                            # sum_c u_row[c] * v_row[c] * w_row[c]
                            errs[i, j, k] = np.sum(u_row * v_row * w_row)
                
                # Target is 1 for all entries
                return np.sum(np.abs(errs - 1))
            
            current_err = get_error(U, V, W)
            if current_err == 0:
                best_rank = r
                best_decomp = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                found = True
                break
            
            # Local Search
            steps = 0
            max_steps = 5000
            no_improve = 0
            max_no_improve = 200
            
            while steps < max_steps and no_improve < max_no_improve:
                if time.time() > deadline:
                    break
                    
                # Pick random variable
                var_idx = random.randint(0, (nm + mp + pn) * r - 1)
                
                if var_idx < nm * r:
                    idx = var_idx // r
                    c = var_idx % r
                    old_val = U[idx, c]
                    new_val = random.choice(candidates)
                    if new_val == old_val:
                        continue
                    U[idx, c] = new_val
                    new_err = get_error(U, V, W)
                    if new_err < current_err:
                        current_err = new_err
                        no_improve = 0
                    else:
                        U[idx, c] = old_val
                        no_improve += 1
                elif var_idx < nm * r + mp * r:
                    idx = (var_idx - nm * r) // r
                    c = (var_idx - nm * r) % r
                    old_val = V[idx, c]
                    new_val = random.choice(candidates)
                    if new_val == old_val:
                        continue
                    V[idx, c] = new_val
                    new_err = get_error(U, V, W)
                    if new_err < current_err:
                        current_err = new_err
                        no_improve = 0
                    else:
                        V[idx, c] = old_val
                        no_improve += 1
                else:
                    idx = (var_idx - nm * r - mp * r) // r
                    c = (var_idx - nm * r - mp * r) % r
                    old_val = W[idx, c]
                    new_val = random.choice(candidates)
                    if new_val == old_val:
                        continue
                    W[idx, c] = new_val
                    new_err = get_error(U, V, W)
                    if new_err < current_err:
                        current_err = new_err
                        no_improve = 0
                    else:
                        W[idx, c] = old_val
                        no_improve += 1
                        
                steps += 1
                if current_err == 0:
                    break
            
            if current_err == 0:
                best_rank = r
                best_decomp = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                found = True
                break
                
            restarts += 1
            
        if found:
            break
            
    return best_decomp
