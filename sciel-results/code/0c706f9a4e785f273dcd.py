"""Tensor decomposition for matrix multiplication 3x3 * 3x8.

Uses the schoolbook (canonical) decomposition of rank n*m*p = 72.
This is the only decomposition guaranteed to be exactly correct.
"""
import time

DIMENSIONS = [3, 3, 8]

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' whose matrices
    (shapes (n*m, rank), (m*p, rank), (p*n, rank)) exactly
    decompose the matrix multiplication tensor for
    (n, m, p) = (3, 3, 8) under the W[k*n+i, r] convention.
    """
    n, m, p = DIMENSIONS
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

if __name__ == "__main__":
    import time
    t0 = time.time()
    result = construct()
    t1 = time.time()
    print(f"Schoolbook decomposition: rank={len(result['U'][0])}, "
          f"U={len(result['U'])}x{len(result['U'][0])}, "
          f"V={len(result['V'])}x{len(result['V'][0])}, "
          f"W={len(result['W'])}x{len(result['W'][0])}, "
          f"time={t1 - t0:.4f}s")
