import time
import numpy as np
from fractions import Fraction

_START_TIME = time.time()
_TIME_LIMIT = 138  # seconds, leave 12s margin for 150s limit

def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition using integer arithmetic."""
    rank = len(U[0]) if U else 0
    
    def to_int2(x):
        v = int(round(x * 2))
        if abs(x * 2 - v) > 1e-9:
            return None
        return v
    
    U2 = [[to_int2(x) for x in row] for row in U]
    V2 = [[to_int2(x) for x in row] for row in V]
    W2 = [[to_int2(x) for x in row] for row in W]
    
    for mat in [U2, V2, W2]:
        for row in mat:
            if any(x is None for x in row):
                return False
    
    U3 = np.array(U2).reshape(n, m, rank)
    V3 = np.array(V2).reshape(m, p, rank)
    W3 = np.array(W2).reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U3[i, j, c] * V3[j, k, c] * W3[k, i, c]
                if s != 4:
                    return False
    return True

def _als_fast(n, m, p, rank, seed, max_iters=400):
    """Fast ALS for tensor decomposition."""
    rng = np.random.default_rng(seed)
    
    U = rng.standard_normal((n * m, rank))
    V = rng.standard_normal((m * p, rank))
    W = rng.standard_normal((p * n, rank))
    
    for c in range(rank):
        nU = np.linalg.norm(U[:, c])
        nV = np.linalg.norm(V[:, c])
        nW = np.linalg.norm(W[:, c])
        if nU > 1e-15: U[:, c] /= nU
        if nV > 1e-15: V[:, c] /= nV
        if nW > 1e-15: W[:, c] /= nW
    
    prev_res = np.inf
    stall = 0
    
    for it in range(max_iters):
        if time.time() - _START_TIME > _TIME_LIMIT - 8:
            return None
        
        # Update U
        for i in range(n):
            for j in range(m):
                Vj = V[j * p:(j + 1) * p, :]
                Wi = W[np.arange(p) * n + i, :]
                M = Vj * Wi
                try:
                    U[i * m + j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                Ui = U[np.arange(n) * m + j, :]
                Wk = W[k * n:(k + 1) * n, :]
                M = Ui * Wk
                try:
                    V[j * p + k, :] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                U_slice = U.reshape(n, m, rank)[i, :, :]
                Vk = V[np.arange(m) * p + k, :]
                M = U_slice * Vk
                try:
                    W[k * n + i, :] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        recon = np.zeros((n, m, p), dtype=np.float64)
        for c in range(rank):
            recon += np.einsum('ij,jk,ki->ijk', U3[:, :, c], V3[:, :, c], W3[:, :, c])
        res = np.max(np.abs(recon - 1.0))
        
        if res < prev_res:
            prev_res = res
            stall = 0
        else:
            stall += 1
        
        if res < 1e-12:
            return U, V, W
        
        if stall > 15:
            U += rng.standard_normal((n * m, rank)) * 0.5
            V += rng.standard_normal((m * p, rank)) * 0.5
            W += rng.standard_normal((p * n, rank)) * 0.5
            stall = 0
    
    return None

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = 2, 4, 7
    best = _build_schoolbook(n, m, p)
    best_rank = n * m * p

    # Try ranks from 30 to 50
    for rank in range(30, 51):
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
        
        num_trials = 20 if rank <= 35 else 10
        for trial in range(num_trials):
            if time.time() - _START_TIME > _TIME_LIMIT - 10:
                break
            seed = rank * 100 + trial * 7
            res = _als_fast(n, m, p, rank, seed)
            if res is not None:
                U, V, W = res
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                U_r = [[round(x * 2) / 2 for x in row] for row in U_list]
                V_r = [[round(x * 2) / 2 for x in row] for row in V_list]
                W_r = [[round(x * 2) / 2 for x in row] for row in W_list]
                if _verify_exact(U_r, V_r, W_r, n, m, p):
                    return {"U": U_r, "V": V_r, "W": W_r}
                
                U_int = [[int(round(x)) for x in row] for row in U_list]
                V_int = [[int(round(x)) for x in row] for row in V_list]
                W_int = [[int(round(x)) for x in row] for row in W_list]
                if _verify_exact(U_int, V_int, W_int, n, m, p):
                    return {"U": U_int, "V": V_int, "W": W_int}
    
    return best
