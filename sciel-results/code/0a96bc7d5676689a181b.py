# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _trivial_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
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

def _compute_error(U, V, W, n, m, p):
    """Compute the sum of squared errors for the decomposition."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                err += (s - 1.0) ** 2
    return err

def _random_search_rank(n, m, p, target_rank, max_iters=5000, seed=42):
    """Try to find a decomposition of given rank using random initialization + local search."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, target_rank) * 0.5
    V = rng.randn(m * p, target_rank) * 0.5
    W = rng.randn(p * n, target_rank) * 0.5
    
    best_err = _compute_error(U, V, W, n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if rng.rand() < 0.33:
            r = rng.randint(target_rank)
            U[:, r] += rng.randn(n * m) * 0.1
        elif rng.rand() < 0.66:
            r = rng.randint(target_rank)
            V[:, r] += rng.randn(m * p) * 0.1
        else:
            r = rng.randint(target_rank)
            W[:, r] += rng.randn(p * n) * 0.1
        
        err = _compute_error(U, V, W, n, m, p)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if best_err < 1e-10:
                break
    
    if best_err < 1e-4:
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        if _verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
            return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Try ranks from 18 upward
    for target_rank in range(18, 25):
        if time.time() > deadline:
            break
        
        # Try multiple seeds for each rank
        for seed_offset in range(5):
            if time.time() > deadline:
                break
            result = _random_search_rank(n, m, p, target_rank, max_iters=500, seed=target_rank + seed_offset)
            if result is not None:
                U, V, W = result
                if _verify_decomposition(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
    
    # Fallback: trivial decomposition
    return _trivial_decomposition(n, m, p)
# EVOLVE-BLOCK-END
