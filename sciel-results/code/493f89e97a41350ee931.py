import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    max_time = 145  # Safety margin from 150s limit
    
    n, m, p = 2, 3, 4
    
    # Target tensor: T[i,j,k] = 1 for all i,j,k
    T = np.ones((n, m, p))
    
    def verify_decomposition(U, V, W, R):
        """Verify that U, V, W with rank R exactly decompose the tensor."""
        U3 = np.array(U).reshape(n, m, R)
        V3 = np.array(V).reshape(m, p, R)
        W3 = np.array(W).reshape(p, n, R)
        A_approx = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.allclose(A_approx, T, atol=1e-9)
    
    def calculate_error(U3, V3, W3):
        A_approx = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.sum((T - A_approx) ** 2)
    
    def als_step(U3, V3, W3, R):
        """Perform one step of Alternating Least Squares."""
        # Update U
        for i in range(n):
            for j in range(m):
                # M is p x R
                M = V3[j] * W3[:, i]
                U3[i, j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
        
        # Update V
        for j in range(m):
            for k in range(p):
                # M is n x R
                M = U3[:, j] * W3[k, :]
                V3[j, k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
        
        # Update W
        for k in range(p):
            for i in range(n):
                # C is m x R
                C = U3[i, :, :] * V3[:, k, :]
                W3[k, i, :] = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
    
    def try_rank(R, max_restarts=30, max_als_iters=100, max_local_steps=3000):
        """Try to find rank R decomposition using ALS and local search."""
        
        for attempt in range(max_restarts):
            if time.time() - start_time > max_time:
                return None
            
            # Random initialization
            U3 = np.random.uniform(-1, 1, (n, m, R))
            V3 = np.random.uniform(-1, 1, (m, p, R))
            W3 = np.random.uniform(-1, 1, (p, n, R))
            
            # Run ALS
            for _ in range(max_als_iters):
                als_step(U3, V3, W3, R)
                
                # Periodically check for convergence to integer/half-integer
                if _ % 10 == 9:
                    # Snap to half-integers
                    U_snap = np.round(U3 * 2) / 2
                    V_snap = np.round(V3 * 2) / 2
                    W_snap = np.round(W3 * 2) / 2
                    
                    # Verify
                    err = calculate_error(U_snap, V_snap, W_snap)
                    
                    if err < 1e-8:
                        # Convert to list
                        return {
                            "U": U_snap.reshape(n*m, R).tolist(),
                            "V": V_snap.reshape(m*p, R).tolist(),
                            "W": W_snap.reshape(p*n, R).tolist()
                        }
            
            # After ALS, try local discrete search
            # We work on the flattened arrays for easier indexing
            U_flat = U3.flatten().copy()
            V_flat = V3.flatten().copy()
            W_flat = W3.flatten().copy()
            
            current_err = calculate_error(U3, V3, W3)
            
            # Local search: try changing one variable to nearest int/half-int
            for step in range(max_local_steps):
                if time.time() - start_time > max_time:
                    return None
                    
                improved = False
                # Randomly select a variable to try
                var_type = np.random.choice(3)
                
                if var_type == 0:
                    idx = np.random.randint(len(U_flat))
                    old_val = U_flat[idx]
                    # Candidates: nearest integer, nearest half-integer
                    cand_int = np.round(old_val)
                    cand_half = np.round(old_val * 2) / 2
                    for cand in [cand_int, cand_half]:
                        if cand == old_val:
                            continue
                        U_flat[idx] = cand
                        U3_tmp = U_flat.reshape(n, m, R)
                        new_err = calculate_error(U3_tmp, V3, W3)
                        if new_err < current_err:
                            current_err = new_err
                            improved = True
                            break
                        else:
                            U_flat[idx] = old_val
                elif var_type == 1:
                    idx = np.random.randint(len(V_flat))
                    old_val = V_flat[idx]
                    cand_int = np.round(old_val)
                    cand_half = np.round(old_val * 2) / 2
                    for cand in [cand_int, cand_half]:
                        if cand == old_val:
                            continue
                        V_flat[idx] = cand
                        V3_tmp = V_flat.reshape(m, p, R)
                        new_err = calculate_error(U3, V3_tmp, W3)
                        if new_err < current_err:
                            current_err = new_err
                            improved = True
                            break
                        else:
                            V_flat[idx] = old_val
                else:
                    idx = np.random.randint(len(W_flat))
                    old_val = W_flat[idx]
                    cand_int = np.round(old_val)
                    cand_half = np.round(old_val * 2) / 2
                    for cand in [cand_int, cand_half]:
                        if cand == old_val:
                            continue
                        W_flat[idx] = cand
                        W3_tmp = W_flat.reshape(p, n, R)
                        new_err = calculate_error(U3, V3, W3_tmp)
                        if new_err < current_err:
                            current_err = new_err
                            improved = True
                            break
                        else:
                            W_flat[idx] = old_val
                
                if not improved:
                    # No improvement in this step, maybe break or continue
                    # To save time, if we get stuck, break
                    if step > 100:
                        break
            
            # Final check
            U_final = U_flat.reshape(n, m, R)
            V_final = V_flat.reshape(m, p, R)
            W_final = W_flat.reshape(p, n, R)
            
            # Snap again just in case
            U_final = np.round(U_final * 2) / 2
            V_final = np.round(V_final * 2) / 2
            W_final = np.round(W_final * 2) / 2
            
            err = calculate_error(U_final, V_final, W_final)
            if err < 1e-8:
                return {
                    "U": U_final.reshape(n*m, R).tolist(),
                    "V": V_final.reshape(m*p, R).tolist(),
                    "W": W_final.reshape(p*n, R).tolist()
                }
        
        return None
    
    # Try ranks from 20 upwards
    for R in [20, 21, 22, 23]:
        if time.time() - start_time > max_time:
            break
        dec = try_rank(R)
        if dec and verify_decomposition(dec["U"], dec["V"], dec["W"], R):
            return dec
    
    # Fallback to schoolbook rank 24
    rank = 24
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
