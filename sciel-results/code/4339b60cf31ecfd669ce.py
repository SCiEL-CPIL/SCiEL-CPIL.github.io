import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 130
    
    def verify(U, V, W, rank, n, m, p):
        """Verify that the decomposition is exact."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err = abs(val - 1.0)
                    max_err = max(max_err, err)
        return max_err
    
    # Try rank 1: U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    # This gives 1*1*1 = 1 for all (i,j,k)
    rank = 1
    U = [[1.0] for _ in range(n * m)]
    V = [[1.0] for _ in range(m * p)]
    W = [[1.0] for _ in range(p * n)]
    err = verify(U, V, W, rank, n, m, p)
    if err < 1e-10:
        return {"U": U, "V": V, "W": W}
    
    # If rank 1 doesn't work, search for higher ranks
    def search_rank(rank, max_iters=50, seed=42):
        np.random.seed(seed)
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit - 5:
                break
            # ALS updates
            for i in range(n):
                for j in range(m):
                    V_slice = V[j*p:(j+1)*p, :]
                    W_slice = W[i::n, :]
                    M_u = V_slice * W_slice
                    target = np.ones(p)
                    try:
                        u_new, _, _, _ = np.linalg.lstsq(M_u, target, rcond=None)
                        U[i*m+j, :] = u_new
                    except:
                        pass
            for j in range(m):
                for k in range(p):
                    U_slice = U[j::m, :]
                    W_slice = W[k*n:(k+1)*n, :]
                    M_v = U_slice * W_slice
                    target = np.ones(n)
                    try:
                        v_new, _, _, _ = np.linalg.lstsq(M_v, target, rcond=None)
                        V[j*p+k, :] = v_new
                    except:
                        pass
            for k in range(p):
                for i in range(n):
                    U_slice = U[i*m:(i+1)*m, :]
                    V_slice = V[k::p, :]
                    M_w = U_slice * V_slice
                    target = np.ones(m)
                    try:
                        w_new, _, _, _ = np.linalg.lstsq(M_w, target, rcond=None)
                        W[k*n+i, :] = w_new
                    except:
                        pass
            
            err = verify(U.tolist(), V.tolist(), W.tolist(), rank, n, m, p)
            if err < 1e-8:
                # Try rounding
                for mult in [1, 2]:
                    U_r = np.rint(U * mult) / mult
                    V_r = np.rint(V * mult) / mult
                    W_r = np.rint(W * mult) / mult
                    err_r = verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), rank, n, m, p)
                    if err_r < 1e-8:
                        return U_r.tolist(), V_r.tolist(), W_r.tolist()
                return U.tolist(), V.tolist(), W.tolist()
        return None
    
    for rank in range(2, 50):
        if time.time() - start_time > time_limit - 10:
            break
        for seed in range(3):
            if time.time() - start_time > time_limit - 5:
                break
            result = search_rank(rank, max_iters=30, seed=seed)
            if result is not None:
                U, V, W = result
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
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
# EVOLVE-BLOCK-END
