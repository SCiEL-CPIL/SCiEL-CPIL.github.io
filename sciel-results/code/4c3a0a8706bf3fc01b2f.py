"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the matmul tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    deadline = time.time() + 100
    
    best_result = None
    best_rank = 60
    
    # Try ranks from 48 to 59
    for rank in range(48, 60):
        if time.time() > deadline - 20:
            break
        for trial in range(3):
            if time.time() > deadline - 20:
                break
            result = _als_decompose(T, nm, mp, pn, rank, max_iters=500, seed=trial)
            if result is not None:
                U, V, W = result
                best_rank = rank
                best_result = (U, V, W)
                break
        if best_result is not None and best_rank < 55:
            break
    
    if best_result is None:
        rank = 60
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_result[0].tolist(), "V": best_result[1].tolist(), "W": best_result[2].tolist()}

def _als_decompose(T, nm, mp, pn, rank, max_iters=500, tol=1e-10, seed=0):
    np.random.seed(seed * 1000 + 42)
    
    U = np.random.randn(nm, rank)
    V = np.random.randn(mp, rank)
    W = np.random.randn(pn, rank)
    
    for it in range(max_iters):
        M_U = np.einsum('jr,kr->jkr', V, W).reshape(mp * pn, rank)
        U_new = np.array([np.linalg.lstsq(M_U, T[i].ravel(), rcond=None)[0] for i in range(nm)])
        
        M_V = np.einsum('ir,kr->ikr', U_new, W).reshape(nm * pn, rank)
        V_new = np.array([np.linalg.lstsq(M_V, T[:, j, :].ravel(), rcond=None)[0] for j in range(mp)])
        
        M_W = np.einsum('ir,jr->ijr', U_new, V_new).reshape(nm * mp, rank)
        W_new = np.array([np.linalg.lstsq(M_W, T[:, :, k].ravel(), rcond=None)[0] for k in range(pn)])
        
        U, V, W = U_new, V_new, W_new
        
        T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
        err = np.linalg.norm(T_approx - T)
        if err < tol:
            return U, V, W
    
    T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
    err = np.linalg.norm(T_approx - T)
    if err < tol:
        return U, V, W
    return None
# EVOLVE-BLOCK-END

