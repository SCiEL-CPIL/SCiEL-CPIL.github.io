"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-10)

def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p

    rng = np.random.default_rng(42)

    for rank in range(best_rank - 1, max(28, best_rank - 30), -1):
        if time.time() > deadline:
            break
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            density = 0.1
            U_rand = rng.standard_normal((n * m, rank)) * (rng.random((n * m, rank)) < density)
            V_rand = rng.standard_normal((m * p, rank)) * (rng.random((m * p, rank)) < density)
            W_rand = rng.standard_normal((p * n, rank)) * (rng.random((p * n, rank)) < density)
            if _verify(U_rand.tolist(), V_rand.tolist(), W_rand.tolist(), n, m, p):
                best_rank = rank
                U, V, W = U_rand.tolist(), V_rand.tolist(), W_rand.tolist()
                break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
