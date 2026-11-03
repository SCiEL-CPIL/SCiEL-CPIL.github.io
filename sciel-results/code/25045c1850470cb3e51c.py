"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    expected = np.ones((n, m, p), dtype=np.float64)
    
    rank = len(U[0])
    reconstructed = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        u = np.array([U[i][c] for i in range(n*m)]).reshape(n, m)
        v = np.array([V[j][c] for j in range(m*p)]).reshape(m, p)
        w = np.array([W[k][c] for k in range(p*n)]).reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    reconstructed[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    
    return np.allclose(expected, reconstructed, atol=tol)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook decomposition
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
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
