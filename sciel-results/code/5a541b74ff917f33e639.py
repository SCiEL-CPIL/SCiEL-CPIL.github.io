"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    r = len(U[0]) if U else 0
    T = np.zeros((n*m, m*p, p*n))
    for t in range(r):
        T += np.einsum('a,b,c->abc', U[:,t], V[:,t], W[:,t])
    T_true = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_true[a, b, c] = 1.0
    return np.allclose(T, T_true, atol=tol)

def _schoolbook(n, m, p):
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

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Build target tensor
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m + j, j*p + k, k*n + i] = 1.0

    def try_lower_rank(r, max_trials=500):
        """Try to find rank-r decomposition."""
        deadline = time.time() + 140
        rng = np.random.default_rng(r * 31 + 7)

        for trial in range(max_trials):
            if time.time() > deadline:
                break

            U = rng.standard_normal((nm, r)) * 0.5
            V = rng.standard_normal((mp, r)) * 0.5
            W = rng.standard_normal((pn, r)) * 0.5

            # Check
            T = np.einsum('at,bt,ct->abc', U, V, W)
            if np.max(np.abs(T - T_target)) < 1e-8:
                return U, V, W
        return None

    # Try rank 29
    result = try_lower_rank(29, max_trials=200)
    if result is not None:
        U, V, W = result
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
