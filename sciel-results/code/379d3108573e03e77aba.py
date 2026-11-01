"""Matrix multiplication 2,2,12 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 12]

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify tensor decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1"""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(U.shape[1]):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_decomposition(n, m, p):
    """Build the known rank-42 Strassen-based decomposition."""
    num_pairs = p // 2
    rank = num_pairs * 7
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    strassen_pattern = [
        {'U': [(0, 1), (3, 1)], 'V': [(0, 0, 1), (1, 1, 1)], 'W': [(0, 0, 1), (1, 1, 1)]},
        {'U': [(2, 1), (3, 1)], 'V': [(0, 0, 1)], 'W': [(0, 1, 1), (1, 1, -1)]},
        {'U': [(0, 1)], 'V': [(0, 1, 1), (1, 1, -1)], 'W': [(1, 0, 1), (1, 1, 1)]},
        {'U': [(3, 1)], 'V': [(1, 0, 1), (0, 0, -1)], 'W': [(0, 0, 1), (0, 1, 1)]},
        {'U': [(0, 1), (1, 1)], 'V': [(1, 1, 1)], 'W': [(0, 0, -1), (1, 0, 1)]},
        {'U': [(2, 1), (0, -1)], 'V': [(0, 0, 1), (0, 1, 1)], 'W': [(1, 1, 1)]},
        {'U': [(1, 1), (3, -1)], 'V': [(1, 0, 1), (1, 1, 1)], 'W': [(0, 0, 1)]},
    ]
    
    for t in range(num_pairs):
        base_c = t * 7
        k0 = 2 * t
        for s in range(7):
            c = base_c + s
            pat = strassen_pattern[s]
            for u_idx, coeff in pat['U']:
                U[u_idx, c] += coeff
            for j, k_loc, coeff in pat['V']:
                global_k = k0 + k_loc
                V[j * p + global_k, c] += coeff
            for k_loc, i, coeff in pat['W']:
                global_k = k0 + k_loc
                W[global_k * n + i, c] += coeff
    return U, V, W

def _try_lower_rank(n, m, p, target_rank, start_time, deadline, seed=42):
    """Attempt to find a decomposition of target_rank using numerical optimization."""
    rng = np.random.RandomState(seed)
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    # Parameterize: flatten U, V, W into a single vector
    # x = [U.flatten(), V.flatten(), W.flatten()]
    x0 = rng.randn(num_u * target_rank + num_v * target_rank + num_w * target_rank) * 0.1
    
    def objective(x):
        U = x[:num_u * target_rank].reshape(num_u, target_rank)
        V = x[num_u * target_rank:num_u * target_rank + num_v * target_rank].reshape(num_v, target_rank)
        W = x[num_u * target_rank + num_v * target_rank:].reshape(num_w, target_rank)
        # Compute tensor: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # We want T[i,j,k] = 1 for all i,j,k
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(target_rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (s - 1.0) ** 2
        return err
    
    # Try a few random restarts
    best_x = None
    best_err = float('inf')
    
    max_restarts = 10
    for r in range(max_restarts):
        if time.time() > deadline - 30:
            break
        x_init = rng.randn(num_u * target_rank + num_v * target_rank + num_w * target_rank) * 0.5
        try:
            res = minimize(objective, x_init, method='L-BFGS-B', 
                          options={'maxiter': 500, 'maxfun': 2000})
            if res.fun < best_err:
                best_err = res.fun
                best_x = res.x
        except Exception:
            pass
    
    if best_x is not None and best_err < 1e-4:
        U = best_x[:num_u * target_rank].reshape(num_u, target_rank)
        V = best_x[num_u * target_rank:num_u * target_rank + num_v * target_rank].reshape(num_v, target_rank)
        W = best_x[num_u * target_rank + num_v * target_rank:].reshape(num_w, target_rank)
        if _verify(U, V, W, n, m, p, tol=1e-4):
            return U, V, W
    return None

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    # First, get the known rank-42 solution as fallback
    U42, V42, W42 = _strassen_decomposition(n, m, p)
    best_rank = 42
    best_U, best_V, best_W = U42, V42, W42
    
    # Try to find lower-rank decompositions
    for target_rank in [36, 32, 28]:
        if time.time() > deadline - 60:
            break
        result = _try_lower_rank(n, m, p, target_rank, start_time, deadline)
        if result is not None:
            best_rank = target_rank
            best_U, best_V, best_W = result
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
