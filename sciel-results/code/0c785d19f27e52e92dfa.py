import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    n_rows_U = n * m  # 15
    n_rows_V = m * p  # 25
    n_rows_W = p * n  # 15
    
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n_rows_U, rank))
        V = np.zeros((n_rows_V, rank))
        W = np.zeros((n_rows_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Precompute target tensor T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) consistent
    # a in [0, n*m), b in [0, m*p), c in [0, p*n)
    T = np.zeros((n_rows_U, n_rows_V, n_rows_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def check_error(U, V, W, r):
        """Check if U,V,W with r columns exactly decompose the matmul tensor."""
        U_t = U.reshape(n, m, r)
        V_t = V.reshape(m, p, r)
        W_t = W.reshape(p, n, r)
        
        # E[i, k, j, l] = sum_c U[i,j,c] * W[k,i,c] * V[l,k,c]
        # Must equal 1 if j==l else 0
        T1 = np.einsum('ijc,kic->ikjc', U_t, W_t)
        E = np.einsum('ikjc,lkc->ikjl', T1, V_t)
        
        D = np.eye(m)
        Target = np.zeros((n, p, m, m))
        Target[:, :, :, :] = D[np.newaxis, np.newaxis, :, :]
        
        error = np.sum((E - Target) ** 2)
        return error

    def try_rank(r, max_restarts=10, max_iters=80):
        """Try to find a rank-r decomposition using ALS."""
        if time.time() > deadline:
            return None
        
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization with small values
            scale = 1.0
            U = np.random.randn(n_rows_U, r) * scale
            V = np.random.randn(n_rows_V, r) * scale
            W = np.random.randn(n_rows_W, r) * scale
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Update U: for each (i,j), solve M_U x = t_U
                for i in range(n):
                    for j in range(m):
                        # M_U[k, l, c] = W[k*n+i, c] * V[l*p+k, c]
                        # t_U[k, l] = 1 if j == l else 0
                        M_U = np.zeros((p * m, r))
                        t_U = np.zeros(p * m)
                        idx = 0
                        for k in range(p):
                            for l in range(m):
                                M_U[idx, :] = W[k * n + i, :] * V[l * p + k, :]
                                if j == l:
                                    t_U[idx] = 1.0
                                idx += 1
                        sol, residuals, rank_s, sv = np.linalg.lstsq(M_U, t_U, rcond=None)
                        U[i * m + j, :] = sol
                
                # Update V: for each (l,k), solve M_V x = t_V
                for l in range(m):
                    for k in range(p):
                        # M_V[i, j, c] = U[i*m+j, c] * W[k*n+i, c]
                        # t_V[i, j] = 1 if j == l else 0
                        M_V = np.zeros((n * m, r))
                        t_V = np.zeros(n * m)
                        idx = 0
                        for i in range(n):
                            for j in range(m):
                                M_V[idx, :] = U[i * m + j, :] * W[k * n + i, :]
                                if j == l:
                                    t_V[idx] = 1.0
                                idx += 1
                        sol, residuals, rank_s, sv = np.linalg.lstsq(M_V, t_V, rcond=None)
                        V[l * p + k, :] = sol
                
                # Update W: for each (k,i), solve M_W x = t_W
                for k in range(p):
                    for i in range(n):
                        # M_W[j, l, c] = U[i*m+j, c] * V[l*p+k, c]
                        # t_W[j, l] = 1 if j == l else 0
                        M_W = np.zeros((m * m, r))
                        t_W = np.zeros(m * m)
                        idx = 0
                        for j in range(m):
                            for l in range(m):
                                M_W[idx, :] = U[i * m + j, :] * V[l * p + k, :]
                                if j == l:
                                    t_W[idx] = 1.0
                                idx += 1
                        sol, residuals, rank_s, sv = np.linalg.lstsq(M_W, t_W, rcond=None)
                        W[k * n + i, :] = sol
                
                # Check convergence
                err = check_error(U, V, W, r)
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if err < 1e-10:
                    break
            
            # After ALS, try to snap to half-integers
            if best_err < 1e-6:
                U_snap = np.round(best_U * 2) / 2
                V_snap = np.round(best_V * 2) / 2
                W_snap = np.round(best_W * 2) / 2
                
                err_snap = check_error(U_snap, V_snap, W_snap, r)
                if err_snap < 1e-8:
                    return U_snap, V_snap, W_snap, r
                
                # Also try integer snap
                U_int = np.round(best_U)
                V_int = np.round(best_V)
                W_int = np.round(best_W)
                err_int = check_error(U_int, V_int, W_int, r)
                if err_int < 1e-8:
                    return U_int, V_int, W_int, r
        
        # Return best found even if not snapped (will be verified by evaluator)
        if best_err < 1e-6:
            return best_U, best_V, best_W, r
        
        return None

    # Search for minimum rank
    # Lower bound: for n×m×p matmul, rank >= max(n*m, m*p, p*n) = max(15,25,15) = 25
    # But better bounds exist. For 3×5×5, let's search from ~40 upward
    # Actually, let's be smarter: start from a reasonable lower bound
    # The trivial lower bound is max(n*m, m*p, p*n) = 25
    # Strassen-like: for square n×n, rank ~ n^2.1... For 3×5×5, try starting at 40
    
    search_start = 30
    search_end = best_rank - 1  # 74
    
    for r in range(search_start, search_end + 1):
        if time.time() > deadline:
            break
        
        result = try_rank(r, max_restarts=8, max_iters=60)
        if result is not None:
            best_U, best_V, best_W, best_rank = result
            break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
