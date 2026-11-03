import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    def build_rank55():
        rank = 55
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        
        # Part A: j=0 (2 x 10 = 20 terms)
        for i in range(n):
            for k in range(p):
                U[i * m + 0, col] = 1
                V[0 * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
        
        # Part B: j=1,2 with Strassen on 5 blocks of 2x2
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            u_11 = 0 * m + 1
            u_12 = 0 * m + 2
            u_21 = 1 * m + 1
            u_22 = 1 * m + 2
            
            v_11 = 1 * p + k0
            v_12 = 1 * p + k1
            v_21 = 2 * p + k0
            v_22 = 2 * p + k1
            
            # M1 = (a11 + a22)(b11 + b22)
            U[u_11, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1; W[w_11, col] += 1
            col += 1
            
            # M2 = (a21 + a22)b11
            U[u_21, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1
            W[w_10, col] += 1; W[w_11, col] += -1
            col += 1
            
            # M3 = a11(b12 - b22)
            U[u_11, col] += 1
            V[v_12, col] += 1; V[v_22, col] += -1
            W[w_01, col] += 1; W[w_11, col] += 1
            col += 1
            
            # M4 = a22(b21 - b11)
            U[u_22, col] += 1
            V[v_21, col] += 1; V[v_11, col] += -1
            W[w_00, col] += 1; W[w_10, col] += 1
            col += 1
            
            # M5 = (a11 + a12)b22
            U[u_11, col] += 1; U[u_12, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += -1; W[w_01, col] += 1
            col += 1
            
            # M6 = (a21 - a11)(b11 + b12)
            U[u_21, col] += 1; U[u_11, col] += -1
            V[v_11, col] += 1; V[v_12, col] += 1
            W[w_11, col] += 1
            col += 1
            
            # M7 = (a12 - a22)(b21 + b22)
            U[u_12, col] += 1; U[u_22, col] += -1
            V[v_21, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1
            col += 1
        
        return U, V, W, rank
    
    def verify(U, V, W, r, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    if abs(val - 1.0) > tol:
                        return False, abs(val - 1.0)
        return True, 0.0
    
    def compute_error(U, V, W, r):
        """Vectorized error computation - returns max absolute error"""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :r], V[j*p + k, :r] * W[k*n + i, :r])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def compute_sse(U, V, W, r):
        """Sum of squared errors for smoother optimization"""
        sse = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :r], V[j*p + k, :r] * W[k*n + i, :r])
                    sse += (val - 1.0) ** 2
        return sse
    
    def polish(U, V, W, r, max_iter=50, tol=1e-12):
        """Final high-precision polish using alternating least squares"""
        for it in range(max_iter):
            if time.time() > deadline - 2:
                break
            
            # Update U
            for idx in range(nm):
                i, j = divmod(idx, m)
                A_mat = np.zeros((p, r))
                for k in range(p):
                    A_mat[k, :] = V[j*p + k, :r] * W[k*n + i, :r]
                b_vec = np.ones(p)
                u_sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                U[idx, :r] = u_sol
            
            # Update V
            for idx in range(mp):
                j, k = divmod(idx, p)
                A_mat = np.zeros((n, r))
                for ii in range(n):
                    A_mat[ii, :] = U[ii*m + j, :r] * W[k*n + ii, :r]
                b_vec = np.ones(n)
                v_sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                V[idx, :r] = v_sol
            
            # Update W
            for idx in range(pn):
                k, i = divmod(idx, n)
                A_mat = np.zeros((m, r))
                for j in range(m):
                    A_mat[j, :] = U[i*m + j, :r] * V[j*p + k, :r]
                b_vec = np.ones(m)
                w_sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W[idx, :r] = w_sol
            
            err = compute_error(U, V, W, r)
            if err < tol:
                break
        
        return U, V, W, compute_error(U, V, W, r)
    
    # Start with known rank-55 solution
    U_best, V_best, W_best, rank_best = build_rank55()
    valid, err = verify(U_best, V_best, W_best, rank_best)
    
    # Try to reduce rank using iterative optimization
    if time.time() < deadline - 20:
        for target_rank in range(rank_best - 1, 40, -1):
            if time.time() > deadline - 15:
                break
            
            found = False
            max_restarts = 10 if target_rank >= 50 else 6
            
            for trial in range(max_restarts):
                if time.time() > deadline - 15:
                    break
                
                np.random.seed(trial * 7919 + target_rank * 13)
                
                # Strategy 1: Truncate and add noise
                U_init = U_best[:, :target_rank].copy()
                V_init = V_best[:, :target_rank].copy()
                W_init = W_best[:, :target_rank].copy()
                
                noise_scale = 0.05
                U_init += noise_scale * np.random.randn(nm, target_rank)
                V_init += noise_scale * np.random.randn(mp, target_rank)
                W_init += noise_scale * np.random.randn(pn, target_rank)
                
                # Strategy 2: Random initialization (for later trials)
                if trial >= max_restarts // 2:
                    U_init = np.random.randn(nm, target_rank) * 0.1
                    V_init = np.random.randn(mp, target_rank) * 0.1
                    W_init = np.random.randn(pn, target_rank) * 0.1
                
                # ALS-style optimization with SSE objective
                U_cur, V_cur, W_cur = U_init.copy(), V_init.copy(), W_init.copy()
                best_err = compute_error(U_cur, V_cur, W_cur, target_rank)
                best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
                
                for iteration in range(100):
                    if time.time() > deadline - 15:
                        break
                    
                    # Update U
                    for idx in range(nm):
                        i, j = divmod(idx, m)
                        A_mat = np.zeros((p, target_rank))
                        for k in range(p):
                            A_mat[k, :] = V_cur[j*p + k, :target_rank] * W_cur[k*n + i, :target_rank]
                        b_vec = np.ones(p)
                        u_sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        U_cur[idx, :target_rank] = u_sol
                    
                    # Update V
                    for idx in range(mp):
                        j, k = divmod(idx, p)
                        A_mat = np.zeros((n, target_rank))
                        for ii in range(n):
                            A_mat[ii, :] = U_cur[ii*m + j, :target_rank] * W_cur[k*n + ii, :target_rank]
                        b_vec = np.ones(n)
                        v_sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        V_cur[idx, :target_rank] = v_sol
                    
                    # Update W
                    for idx in range(pn):
                        k, i = divmod(idx, n)
                        A_mat = np.zeros((m, target_rank))
                        for j in range(m):
                            A_mat[j, :] = U_cur[i*m + j, :target_rank] * V_cur[j*p + k, :target_rank]
                        b_vec = np.ones(m)
                        w_sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        W_cur[idx, :target_rank] = w_sol
                    
                    # Check error
                    err_val = compute_error(U_cur, V_cur, W_cur, target_rank)
                    
                    if err_val < best_err:
                        best_err = err_val
                        best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
                    
                    if best_err < 1e-8:
                        # Polish
                        U_cur, V_cur, W_cur, final_err = polish(
                            U_cur, V_cur, W_cur, target_rank, 
                            max_iter=30, tol=1e-12
                        )
                        if final_err < 1e-10:
                            found = True
                            U_best, V_best, W_best, rank_best = U_cur, V_cur, W_cur, target_rank
                            break
                    elif iteration % 20 == 19 and best_err > 0.5:
                        # Stuck, break early
                        break
                
                if found:
                    break
            
            if found:
                break
    
    # Final verification and conversion
    final_err = compute_error(U_best, V_best, W_best, rank_best)
    if final_err > 1e-6:
        # Fall back to rank-55 if optimization failed
        U_best, V_best, W_best, rank_best = build_rank55()
    
    U_list = U_best.tolist()
    V_list = V_best.tolist()
    W_list = W_best.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
