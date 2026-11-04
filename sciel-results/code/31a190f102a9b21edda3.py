import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank_seed = n * m * p  # 96
    
    # Schoolbook solution
    U0 = np.zeros((n*m, rank_seed))
    V0 = np.zeros((m*p, rank_seed))
    W0 = np.zeros((p*n, rank_seed))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank_seed
    
    def compute_error(U, V, W, R):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :R], V[j*p+k, :R] * W[k*n+i, :R])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def als_step(U, V, W, R, max_iters=50, tol=1e-12):
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        prev_err = float('inf')
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            for i in range(n):
                for j in range(m):
                    B = np.zeros((p, R))
                    for k in range(p):
                        B[k, :] = V[j*p+k, :R] * W[k*n+i, :R]
                    try:
                        U[i*m+j, :R] = np.linalg.lstsq(B, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, R))
                    for i in range(n):
                        B[i, :] = U[i*m+j, :R] * W[k*n+i, :R]
                    try:
                        V[j*p+k, :R] = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            for k in range(p):
                for i in range(n):
                    B = np.zeros((m, R))
                    for j in range(m):
                        B[j, :] = U[i*m+j, :R] * V[j*p+k, :R]
                    try:
                        W[k*n+i, :R] = np.linalg.lstsq(B, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            if it % 5 == 4:
                err = compute_error(U, V, W, R)
                if err < tol:
                    break
                if abs(err - prev_err) < 1e-14:
                    break
                prev_err = err
        
        return U, V, W

    def try_round(U, V, W, R):
        U_r = np.round(2 * U) / 2
        V_r = np.round(2 * V) / 2
        W_r = np.round(2 * W) / 2
        
        err = compute_error(U_r, V_r, W_r, R)
        if err < 1e-9:
            return U_r, V_r, W_r, True
        return None, None, None, False

    # Try ranks from 48 to 95
    for R in range(48, rank_seed):
        if time.time() > deadline - 10:
            break
        
        found = False
        num_restarts = 5
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(12345 + R * 10 + restart)
            
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            U, V, W = als_step(U, V, W, R, max_iters=30)
            
            err = compute_error(U, V, W, R)
            
            if err < 1e-10:
                U_r, V_r, W_r, success = try_round(U, V, W, R)
                if success:
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = R
                    found = True
                    break
        
        if found:
            break

    final_err = compute_error(best_U, best_V, best_W, best_rank)
    if final_err > 1e-9:
        best_U, best_V, best_W = U0, V0, W0
        best_rank = rank_seed

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
