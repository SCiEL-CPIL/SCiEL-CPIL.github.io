"""Search for rank-minimal exact tensor decomposition of 3x4 * 4x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the trivial schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify_tensor(U, V, W, n, m, p):
    """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all (i,j,k)."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _try_random_rank(rank, n, m, p, time_budget, seed_base=0):
    """Try to find a valid decomposition of given rank via random search."""
    start = time.time()
    rng = np.random.RandomState(seed_base)
    values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    attempts = 0
    while time.time() - start < time_budget:
        U = rng.choice(values, size=(n * m, rank))
        V = rng.choice(values, size=(m * p, rank))
        W = rng.choice(values, size=(p * n, rank))
        if _verify_tensor(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        attempts += 1
        if attempts % 10000 == 0:
            break  # Safety
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    best = _schoolbook()
    best_rank = n * m * p
    
    # Quick search for slightly lower ranks
    for rank in range(best_rank - 1, max(20, best_rank - 5), -1):
        if time.time() >= deadline:
            break
        remaining = deadline - time.time()
        if remaining < 2:
            break
        result = _try_random_rank(rank, n, m, p, min(remaining, 10), seed_base=rank * 7919)
        if result is not None:
            best = result
            best_rank = rank
            break
    
    return best
# EVOLVE-BLOCK-END
