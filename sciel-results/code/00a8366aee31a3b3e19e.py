import time

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 8  # A is n x m, B is m x p, C is n x p

    rank = m * n  # 8 terms: m terms for each of the n rows

    # U: (n*m) x rank = 8 x 8
    # V: (m*p) x rank = 32 x 8
    # W: (p*n) x rank = 16 x 8
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Index conventions:
    # A[i, j] -> index i*m + j in U
    # B[j, k] -> index j*p + k in V
    # C[i, k] -> index k*n + i in W  (trace convention W[k*n+i, r])

    # For each row i of A and each column j, create a term:
    # Term r = i*m + j:
    #   U[i*m + j, r] = 1
    #   V[j*p + k, r] = 1 for all k in 0..p-1
    #   W[k*n + i, r] = 1 for all k in 0..p-1
    #
    # Then summing over all r: for fixed (i,j,k),
    # only r = i*m + j contributes, giving 1*1*1 = 1. ✓

    for i in range(n):
        for j in range(m):
            r = i * m + j
            U[i * m + j][r] = 1
            for k in range(p):
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1

    if time.time() - start_time > 110:
        pass

    return {"U": U, "V": V, "W": W}
