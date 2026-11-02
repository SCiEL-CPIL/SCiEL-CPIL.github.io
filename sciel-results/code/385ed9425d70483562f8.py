"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START

def _verify_fast(U, V, W, n, m, p):
    """Fast verification using NumPy."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_r = U_arr.reshape(n, m, rank)
    V_r = V_arr.reshape(m, p, rank)
    W_r = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    
    return np.allclose(T, 1.0, atol=1e-5)

def _verify_slow(U, V, W, n, m, p):
    """Slow but reliable verification."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(val - 1) > 1e-5:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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

def _polish_decomposition(U, V, W, n, m, p, rank, time_limit=5.0):
    """Polish the decomposition using gradient descent to minimize error."""
    start_time = time.time()
    U_flat = np.array(U, dtype=np.float64).flatten()
    V_flat = np.array(V, dtype=np.float64).flatten()
    W_flat = np.array(W, dtype=np.float64).flatten()
    x = np.concatenate([U_flat, V_flat, W_flat])
    
    def objective(x):
        if time.time() - start_time > time_limit:
            return 1e10
        U_f = x[:n*m*rank].reshape(n, m, rank)
        V_f = x[n*m*rank : n*m*rank + m*p*rank].reshape(m, p, rank)
        W_f = x[n*m*rank + m*p*rank :].reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_f, V_f, W_f)
        return np.sum((T - 1.0)**2)
    
    try:
        res = minimize(objective, x, method='L-BFGS-B', options={'maxiter': 50, 'disp': False, 'maxfun': 1000})
        if res.success or objective(res.x) < 1e-4:
            U_res = res.x[:n*m*rank].reshape(n*m, rank).tolist()
            V_res = res.x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank).tolist()
            W_res = res.x[n*m*rank + m*p*rank :].reshape(p*n, rank).tolist()
            return U_res, V_res, W_res
    except:
        pass
    return None

def _als_decomposition(n, m, p, rank, iters=100, seed=0):
    """Attempt to find a decomposition of given rank using ALS."""
    rng = np.random.RandomState(seed)
    
    # Initialize with random values, scaled to 1
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    for it in range(iters):
        # Update U
        for i in range(n):
            for j in range(m):
                M = (V_r[j, :, :] * W_r[:, i, :]).T
                try:
                    U_r[i, j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                B = (U_r[:, j, :] * W_r[k, :, :]).T
                try:
                    V_r[j, k, :] = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                C = (U_r[i, :, :] * V_r[:, k, :]).T
                try:
                    W_r[k, i, :] = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        U = U_r.reshape(n * m, rank)
        V = V_r.reshape(m * p, rank)
        W = W_r.reshape(p * n, rank)
        
        # Early check
        if _verify_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            break
    
    # Polish
    polished = _polish_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p, rank)
    if polished:
        U, V, W = polished
    
    # Verify
    if _verify_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0
    
    # Start with schoolbook
    best = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try ranks from lower bound upward
    # Known rank for 5x5x5 is 19. 4x5x5 should be <= 19.
    ranks_to_try = list(range(19, 35)) + list(range(35, 60, 2))
    
    for target_rank in ranks_to_try:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
            
        max_seeds = 20 if target_rank < 25 else 10
        for seed in range(max_seeds):
            if time.time() > deadline:
                break
            
            result = _als_decomposition(n, m, p, target_rank, iters=50, seed=seed)
            if result is not None:
                U_rounded = [[round(2 * x) / 2 for x in row] for row in result['U']]
                V_rounded = [[round(2 * x) / 2 for x in row] for row in result['V']]
                W_rounded = [[round(2 * x) / 2 for x in row] for row in result['W']]
                
                if _verify_slow(U_rounded, V_rounded, W_rounded, n, m, p):
                    if target_rank < best_rank:
                        best = {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                        best_rank = target_rank
                        if best_rank <= 20:
                            return best
    
    return best
# EVOLVE-BLOCK-END
