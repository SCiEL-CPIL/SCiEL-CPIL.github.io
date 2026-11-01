import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    def verify(U, V, W, rank):
        """Check if the decomposition is exact using vectorized operations."""
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        max_err = np.max(np.abs(T - 1))
        return max_err < 1e-8

    def tensor_error(U, V, W, rank):
        """Compute max absolute error of the tensor reconstruction."""
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        return np.max(np.abs(T - 1))

    # Precompute M1, M2, M3
    M1 = np.zeros((n*m, m*p * p*n))
    M2 = np.zeros((m*p, p*n * n*m))
    M3 = np.zeros((p*n, n*m * m*p))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                col1 = (j*p+k)*(p*n) + i*n+k
                M1[a, col1] = 1
                
                b = j*p+k
                col2 = (i*n+k)*(n*m) + i*m+j
                M2[b, col2] = 1
                
                c = k*n+i
                col3 = (i*m+j)*(m*p) + j*p+k
                M3[c, col3] = 1

    def als_solve_vectorized(U, V, W, rank, max_iters=500, tol=1e-10):
        """Vectorized Alternating Least Squares."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            V_flat = V.T # (rank, mp)
            W_flat = W.T # (rank, np)
            K_VW = (V_flat[:, :, None] * W_flat[:, None, :]).reshape(rank, m*p * p*n)
            try:
                U = np.linalg.lstsq(K_VW.T, M1, rcond=None)[0]
            except:
                return None
                
            # Update V
            W_flat = W.T # (rank, np)
            U_flat = U.T # (rank, nm)
            W_3d = W_flat.reshape(rank, p, n)
            U_3d = U_flat.reshape(rank, n, m)
            K_WU_3d = W_3d[:, :, :, None] * U_3d[:, None, :, :]
            K_WU = K_WU_3d.reshape(rank, p*n * n*m)
            try:
                V = np.linalg.lstsq(K_WU.T, M2, rcond=None)[0]
            except:
                return None
                
            # Update W
            U_flat = U.T # (rank, nm)
            V_flat = V.T # (rank, mp)
            U_3d = U_flat.reshape(rank, n, m)
            V_3d = V_flat.reshape(rank, m, p)
            K_UV_3d = U_3d[:, :, :, None] * V_3d[:, None, :, :]
            K_UV = K_UV_3d.reshape(rank, n*m * m*p)
            try:
                W = np.linalg.lstsq(K_UV.T, M3, rcond=None)[0]
            except:
                return None
                
            err = tensor_error(U, V, W, rank)
            if err < tol:
                return U, V, W
                
        return None

    def round_to_half_int(X):
        """Round to nearest half-integer."""
        return np.round(X * 2) / 2

    rng = np.random.default_rng(42)
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_schoolbook
    
    # Try ranks from 50 down to 30
    # If a lower rank exists, we want to find it.
    # If not, we return the best found (which might be schoolbook).
    for r in range(50, 29, -1):
        if time.time() > deadline:
            break
        
        found = False
        # Try multiple random initializations
        # Increase attempts for lower ranks? No, just keep it consistent but fast.
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            U = rng.standard_normal((n*m, r))
            V = rng.standard_normal((m*p, r))
            W = rng.standard_normal((p*n, r))
            
            result = als_solve_vectorized(U, V, W, r)
            if result is None:
                continue
            
            U_sol, V_sol, W_sol = result
            err = tensor_error(U_sol, V_sol, W_sol, r)
            
            if err < 1e-8:
                U_r = round_to_half_int(U_sol)
                V_r = round_to_half_int(V_sol)
                W_r = round_to_half_int(W_sol)
                
                if verify(U_r, V_r, W_r, r):
                    best_rank = r
                    best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    found = True
                    break
        
        if found:
            # If we found a solution for r, we can try to go lower immediately
            # or just break if we assume we found the minimal one in this range?
            # No, we should continue to check lower ranks.
            # But if we found 50, maybe 49 is possible.
            pass

    return best_result
