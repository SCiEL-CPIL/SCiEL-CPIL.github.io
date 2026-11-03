"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]
import numpy as np
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
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

    # Target tensor T_target[i, j, k] = 1 for all i, j, k
    # We represent T_target as a 3D array of shape (n, m, p)
    T_target = np.ones((n, m, p))
    
    def compute_error(U, V, W):
        # T_rec[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Using einsum for speed
        # U shape: (n*m, R), V shape: (m*p, R), W shape: (p*n, R)
        # We need to map indices carefully.
        # Let's reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        
        # T_rec[i, j, k] = sum_r U_r[i, j, r] * V_r[j, k, r] * W_r[k, i, r]
        # einsum: 'ijr,jkr,ki r->ijk'
        T_rec = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.linalg.norm(T_rec - T_target)

    def als_step(U, V, W, max_iter=100, tol=1e-9):
        # Alternating Least Squares
        # Update U
        # For each i:
        # b_i = T_target[i, :, :] (flattened to m*p)
        # A_i[j*p+k, r] = V[j, r] * W[k, r]
        # Solve A_i.T @ u_i = b_i
        
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        R = U.shape[1]
        
        prev_err = np.inf
        for it in range(max_iter):
            # Update U
            new_U = np.zeros_like(U)
            for i in range(n):
                b = T_target[i, :, :].flatten() # size m*p
                # A size (m*p, R)
                A = np.zeros((m * p, R))
                for j in range(m):
                    for k in range(p):
                        A[j * p + k, :] = V_r[j, :, :] * W_r[k, i, :]
                # Solve A.T @ u = b
                try:
                    u_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                    new_U[i * m : (i+1) * m, :] = u_sol
                except:
                    break
                    
            U = new_U
            U_r = U.reshape(n, m, -1)
            
            # Update V
            new_V = np.zeros_like(V)
            for j in range(m):
                b = T_target[:, j, :].flatten() # size n*p
                # A size (n*p, R)
                A = np.zeros((n * p, R))
                for i in range(n):
                    for k in range(p):
                        A[i * p + k, :] = U_r[i, j, :] * W_r[k, i, :]
                try:
                    v_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                    new_V[j * p : (j+1) * p, :] = v_sol
                except:
                    break
                    
            V = new_V
            V_r = V.reshape(m, p, -1)
            
            # Update W
            new_W = np.zeros_like(W)
            for k in range(p):
                b = T_target[:, :, k].flatten() # size n*m
                # A size (n*m, R)
                A = np.zeros((n * m, R))
                for i in range(n):
                    for j in range(m):
                        A[i * m + j, :] = U_r[i, j, :] * V_r[j, k, :]
                try:
                    w_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                    new_W[k * n : (k+1) * n, :] = w_sol
                except:
                    break
                    
            W = new_W
            W_r = W.reshape(p, n, -1)
            
            err = compute_error(U, V, W)
            if abs(prev_err - err) < tol:
                break
            prev_err = err
            
        return U, V, W, err

    start_time = time.time()
    deadline = start_time + 115 # Safety margin

    best_rank = rank_school
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()

    # Ranks to try. We want minimal rank.
    # Let's try from 45 to 75.
    # If we find a solution at 50, we continue to check 49, 48... 
    # But we are iterating 45, 46, 47...
    # So if 45 fails, 46 fails, 47 succeeds, we record 47.
    # Then we check 48? No, 48 is worse.
    # We should stop if we find a solution?
    # Yes, because we are checking in ascending order. The first success is the minimum rank found in this range.
    # However, if 45 fails due to bad init, but 46 succeeds, we return 46.
    # What if 45 works with a different init?
    # We try multiple inits per rank.
    
    ranks_to_try = list(range(45, rank_school + 1))
    
    for R in ranks_to_try:
        if time.time() > deadline:
            break
            
        found = False
        # Try a few different initializations
        # 1. Prefix init
        # 2. Random init
        
        # Prefix Init
        U_init = U0[:, :R].copy()
        V_init = V0[:, :R].copy()
        W_init = W0[:, :R].copy()
        
        U_curr, V_curr, W_curr, err = als_step(U_init, V_init, W_init, max_iter=200)
        if err < 1e-8:
            best_rank = R
            best_U, best_V, best_W = U_curr, V_curr, W_curr
            found = True
            break
        
        # Random Inits
        for restart in range(5):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + restart * 100 + R)
            U_rand = np.random.randn(n * m, R) * 0.1
            V_rand = np.random.randn(m * p, R) * 0.1
            W_rand = np.random.randn(p * n, R) * 0.1
            
            U_curr, V_curr, W_curr, err = als_step(U_rand, V_rand, W_rand, max_iter=200)
            if err < 1e-8:
                best_rank = R
                best_U, best_V, best_W = U_curr, V_curr, W_curr
                found = True
                break
        
        if found:
            # Since we are iterating R in ascending order, this is the minimal rank found so far.
            # We can break to save time, as lower ranks were already attempted.
            break

    if best_rank < rank_school:
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
    else:
        U_out = U0.tolist()
        V_out = V0.tolist()
        W_out = W0.tolist()

    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
