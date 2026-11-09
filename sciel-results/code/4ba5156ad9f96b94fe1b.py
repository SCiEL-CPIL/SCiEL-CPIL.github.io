"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U = np.zeros((nm, rank_school))
    V = np.zeros((mp, rank_school))
    W = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_school
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Build list of support indices: (i_idx, j_idx, k_idx) where tensor value is 1
    support = []
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i_idx = a*m + b
                j_idx = b*p + c
                k_idx = c*n + a
                support.append((i_idx, j_idx, k_idx))
                
    # Pre-compute support sets for faster gradient calculation if needed, 
    # but for now we iterate directly.
    
    # Try to find lower rank decompositions
    # We iterate rank downwards.
    for r in range(rank_school - 1, max(50, rank_school - 20), -1):
        if time.time() > deadline:
            break
        
        found = False
        # Try multiple random initializations
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            U_try = np.random.randn(nm, r) * 0.1
            V_try = np.random.randn(mp, r) * 0.1
            W_try = np.random.randn(pn, r) * 0.1
            
            # Scale to encourage non-zero values
            U_try *= 1.0
            V_try *= 1.0
            W_try *= 1.0
            
            # Gradient Descent
            lr = 0.001
            # We use a sample of off-support points to keep computation low
            # Total off-support is huge, but we can sample
            # Let's try 200 random off-support points per step
            sample_size = 200
            
            for step in range(200):
                if time.time() > deadline:
                    break
                
                # Calculate gradients
                grad_U = np.zeros_like(U_try)
                grad_V = np.zeros_like(V_try)
                grad_W = np.zeros_like(W_try)
                
                # 1. Support points (must be 1)
                for (i_idx, j_idx, k_idx) in support:
                    u_row = U_try[i_idx]
                    v_row = V_try[j_idx]
                    w_row = W_try[k_idx]
                    
                    # T_ijk = sum_r U[i,r] V[j,r] W[k,r]
                    t_val = np.dot(u_row * v_row, w_row)
                    err = t_val - 1.0
                    
                    # dL/dU[i,r] = -2 * err * V[j,r] * W[k,r]
                    # dL/dV[j,r] = -2 * err * U[i,r] * W[k,r]
                    # dL/dW[k,r] = -2 * err * U[i,r] * V[j,r]
                    
                    grad_U[i_idx] -= 2.0 * err * (v_row * w_row)
                    grad_V[j_idx] -= 2.0 * err * (u_row * w_row)
                    grad_W[k_idx] -= 2.0 * err * (u_row * v_row)
                
                # 2. Off-support points (must be 0)
                # Sample random indices
                for _ in range(sample_size):
                    ii = np.random.randint(0, nm)
                    jj = np.random.randint(0, mp)
                    kk = np.random.randint(0, pn)
                    
                    # Check if this is actually off-support
                    # Support condition: ii = a*m+b, jj = b*p+c, kk = c*n+a
                    a, b = divmod(ii, m)
                    b2, c = divmod(jj, p)
                    c2, a2 = divmod(kk, n)
                    
                    if b != b2 or c != c2 or a != a2:
                        u_row = U_try[ii]
                        v_row = V_try[jj]
                        w_row = W_try[kk]
                        t_val = np.dot(u_row * v_row, w_row)
                        
                        # L = t_val^2, dL/dU = 2 * t_val * V[j,r] * W[k,r]
                        grad_U[ii] += 2.0 * t_val * (v_row * w_row)
                        grad_V[jj] += 2.0 * t_val * (u_row * w_row)
                        grad_W[kk] += 2.0 * t_val * (u_row * v_row)
                
                # Update
                U_try -= lr * grad_U
                V_try -= lr * grad_V
                W_try -= lr * grad_W
                
                # Check convergence
                # Calculate max error on support
                max_err = 0.0
                for (i_idx, j_idx, k_idx) in support:
                    t_val = np.dot(U_try[i_idx, :] * V_try[j_idx, :], W_try[k_idx, :])
                    err = abs(t_val - 1.0)
                    if err > max_err:
                        max_err = err
                
                # Check off-support error on a sample
                max_off_err = 0.0
                for _ in range(50):
                    ii = np.random.randint(0, nm)
                    jj = np.random.randint(0, mp)
                    kk = np.random.randint(0, pn)
                    a, b = divmod(ii, m)
                    b2, c = divmod(jj, p)
                    c2, a2 = divmod(kk, n)
                    if b != b2 or c != c2 or a != a2:
                        t_val = np.dot(U_try[ii, :] * V_try[jj, :], W_try[kk, :])
                        if abs(t_val) > max_off_err:
                            max_off_err = abs(t_val)
                
                if max_err < 1e-8 and max_off_err < 1e-8:
                    # Verify with a larger sample to be safe
                    valid = True
                    for _ in range(500):
                        ii = np.random.randint(0, nm)
                        jj = np.random.randint(0, mp)
                        kk = np.random.randint(0, pn)
                        a, b = divmod(ii, m)
                        b2, c = divmod(jj, p)
                        c2, a2 = divmod(kk, n)
                        if b != b2 or c != c2 or a != a2:
                            t_val = np.dot(U_try[ii, :] * V_try[jj, :], W_try[kk, :])
                            if abs(t_val) > 1e-7:
                                valid = False
                                break
                    
                    if valid:
                        best_U, best_V, best_W = U_try, V_try, W_try
                        best_rank = r
                        found = True
                        break
            
            if found:
                break
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
