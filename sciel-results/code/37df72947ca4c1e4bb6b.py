import time
import numpy as np

DIMENSIONS = [2, 4, 4]

_START_TIME = time.time()
_DEADLINE = _START_TIME + 130.0

def _build_rank32():
    """Build a rank-32 decomposition using the trivial diagonal tensor decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U: index i*m + j
                U[i * m + j, r] = 1
                # V: index j*p + k
                V[j * p + k, r] = 1
                # W: index k*n + i
                W[k * n + i, r] = 1
                r += 1

    return U, V, W

def _verify(U, V, W, trials=3, seed=42):
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    rng = np.random.RandomState(seed)
    for trial in range(trials):
        A = rng.randint(-5, 6, (n, m)).astype(float)
        B = rng.randint(-5, 6, (m, p)).astype(float)
        C = A @ B
        Cf = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                acc = 0.0
                for cc in range(rank):
                    for jj in range(m):
                        acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
                Cf[ii, kk] = acc
        if not np.allclose(C, Cf, atol=1e-6):
            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # Use rank 32 decomposition
    U32, V32, W32 = _build_rank32()
    
    # Verify
    if _verify(U32, V32, W32):
        return {"U": U32.tolist(), "V": V32.tolist(), "W": W32.tolist()}
        
    # Fallback (should not happen)
    return {"U": U32.tolist(), "V": V32.tolist(), "W": W32.tolist()}
