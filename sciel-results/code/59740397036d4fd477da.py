import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
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
    
    def compute_error_fast(U, V, W, r):
        """Vectorized error computation"""
        err_max = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U[i*m + j, :]
                    v_row = V[j*p + k, :]
                    w_row = W[k*n + i, :]
                    val = np.dot(u_row, v_row * w_row)
                    err_max = max(err_max, abs(val - 1.0))
        return err_max
    
    def als_optimize_fast(U_init, V_init, W_init, target_rank, max_iters=300, tol=1e-9):
        """Faster ALS with precomputed structure"""
        U_cur = U_init.copy()
        V_cur = V_init.copy()
        W_cur = W_init.copy()
        best_err = float('inf')
        best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
        
        # Precompute index mappings
        U_idx = [(i, j) for i in range(n) for j in range(m)]
        V_idx = [(j, k) for j in range(m) for k in range(p)]
        W_idx = [(k, i) for k in range(p) for i in range(n)]
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each (i,j), solve min ||A_u @ u - 1||^2
            # where A_u[k, r] = V[j*p+k, r] * W[k*n+i, r]
            for idx, (i, j) in enumerate(U_idx):
                # Build A matrix: shape (p, target_rank)
                A = np.empty((p, target_rank))
                for k in range(p):
                    A[k, :] = V_cur[j*p + k, :] * W_cur[k*n + i, :]
                b = np.ones(p)
                try:
                    u_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U_cur[idx, :] = u_sol
                except:
                    pass
            
            # Update V: for each (j,k), solve min ||A_v @ v - 1||^2
            for idx, (j, k) in enumerate(V_idx):
                A = np.empty((n, target_rank))
                for i in range(n):
                    A[i, :] = U_cur[i*m + j, :] * W_cur[k*n + i, :]
                b = np.ones(n)
                try:
                    v_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V_cur[idx, :] = v_sol
                except:
                    pass
            
            # Update W: for each (k,i), solve min ||A_w @ w - 1||^2
            for idx, (k, i) in enumerate(W_idx):
                A = np.empty((m, target_rank))
                for j in range(m):
                    A[j, :] = U_cur[i*m + j, :] * V_cur[j*p + k, :]
                b = np.ones(m)
                try:
                    w_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W_cur[idx, :] = w_sol
                except:
                    pass
            
            # Check error
            err_val = compute_error_fast(U_cur, V_cur, W_cur, target_rank)
            
            if err_val < best_err:
                best_err = err_val
                best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
                if err_val < tol:
                    break
            
            if err_val < 1e-12:
                break
        
        return best_U, best_V, best_W, best_err
    
    # Start with known rank-55 solution
    U_best, V_best, W_best, rank_best = build_rank55()
    
    # Try to reduce rank using iterative optimization
    if time.time() < deadline - 20:
        for target_rank in range(rank_best - 1, 45, -1):
            if time.time() > deadline - 15:
                break
            
            found = False
            max_restarts = 15 if target_rank >= 50 else 8
            
            for trial in range(max_restarts):
                if time.time() > deadline - 10:
                    break
                
                np.random.seed(trial * 7919 + target_rank * 13)
                
                # Strategy 1: Truncate rank-55 and add noise
                U_init = U_best[:, :target_rank].copy()
                V_init = V_best[:, :target_rank].copy()
                W_init = W_best[:, :target_rank].copy()
                
                # Add random noise to help escape local minima
                noise_scale = 0.03
                U_init += noise_scale * np.random.randn(nm, target_rank)
                V_init += noise_scale * np.random.randn(mp, target_rank)
                W_init += noise_scale * np.random.randn(pn, target_rank)
                
                # Run ALS optimization
                U_cur, V_cur, W_cur, best_err = als_optimize_fast(
                    U_init, V_init, W_init, target_rank, max_iters=200
                )
                
                if best_err < 1e-7:
                    U_best, V_best, W_best, rank_best = U_cur, V_cur, W_cur, target_rank
                    found = True
                    break
                elif best_err < 1e-4:
                    # Good enough to try more iterations with tighter tolerance
                    U_cur2, V_cur2, W_cur2, best_err2 = als_optimize_fast(
                        U_cur, V_cur, W_cur, target_rank, max_iters=300, tol=1e-10
                    )
                    if best_err2 < 1e-7:
                        U_best, V_best, W_best, rank_best = U_cur2, V_cur2, W_cur2, target_rank
                        found = True
                        break
                
                # Strategy 2: Random initialization with structure
                if trial % 3 == 2:
                    U_rand = np.random.randn(nm, target_rank) * 0.1
                    V_rand = np.random.randn(mp, target_rank) * 0.1
                    W_rand = np.random.randn(pn, target_rank) * 0.1
                    
                    U_cur, V_cur, W_cur, best_err = als_optimize_fast(
                        U_rand, V_rand, W_rand, target_rank, max_iters=150
                    )
                    
                    if best_err < 1e-7:
                        U_best, V_best, W_best, rank_best = U_cur, V_cur, W_cur, target_rank
                        found = True
                        break
            
            if found:
                continue  # Try next lower rank
            else:
                break  # Can't reduce further
    
    # Final verification
    err = compute_error_fast(U_best, V_best, W_best, rank_best)
    
    U_list = U_best.tolist()
    V_list = V_best.tolist()
    W_list = W_best.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
