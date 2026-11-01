"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Tensor entry T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should equal 1 if this is the "canonical" entry, 0 otherwise
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
    # in the standard decomposition. Let me reconsider.
    # The tensor for matmul (n x m) * (m x p) -> (n x p) has:
    # T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # The value is 1 (it's the identity tensor in the appropriate sense)
    # Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # For the trivial decomposition, each term picks one (i,j,k).
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

def _random_search_rank(n, m, p, target_rank, max_iters=5000, seed=42):
    """Try to find a decomposition of given rank using random initialization + local search."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, target_rank)
    V = rng.randn(m * p, target_rank)
    W = rng.randn(p * n, target_rank)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                    err += (s - 1.0) ** 2
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if best_err < 1e-12:
            break
        # Random perturbation
        if rng.rand() < 0.3:
            r = rng.randint(target_rank)
            U[:, r] += rng.randn(n * m) * 0.1
        elif rng.rand() < 0.6:
            r = rng.randint(target_rank)
            V[:, r] += rng.randn(m * p) * 0.1
        else:
            r = rng.randint(target_rank)
            W[:, r] += rng.randn(p * n) * 0.1
    
    # Try to round to integers/half-integers
    if best_err < 1e-6:
        # Round each entry to nearest 0.5
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        # Verify
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_rounded[i * m + j], V_rounded[j * p + k] * W_rounded[k * n + i])
                    if abs(s - 1.0) > 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin out of 120s
    
    n, m, p = DIMENSIONS
    # n=2, m=3, p=4, trivial rank=24
    
    # Try ranks from 18 upward
    for target_rank in range(18, 25):
        if time.time() > deadline:
            break
        
        result = _random_search_rank(n, m, p, target_rank, max_iters=2000, seed=target_rank)
        if result is not None:
            U, V, W = result
            # Final verification
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Fallback: trivial decomposition
    return _trivial_decomposition(n, m, p)
# EVOLVE-BLOCK-END
