DIMENSIONS = [2, 4, 8]

def construct():
    """
    Tensor decomposition for (2x4) * (4x8) -> (2x8).
    Returns U (8xrank), V (32xrank), W (16xrank).
    Uses schoolbook decomposition with rank = n*m*p = 64.
    """
    import time
    t0 = time.time()
    deadline = t0 + 140

    n = DIMENSIONS[0]
    m = DIMENSIONS[1]
    p = DIMENSIONS[2]
    n_u = n * m  # 8
    n_v = m * p  # 32
    n_w = p * n  # 16
    rank = n * m * p  # 64

    U = [[0] * rank for _ in range(n_u)]
    V = [[0] * rank for _ in range(n_v)]
    W = [[0] * rank for _ in range(n_w)]

    for i in range(n):
        for j in range(m):
            for k in range(p):
                r_idx = i * m * p + j * p + k
                U[i * m + j][r_idx] = 1
                V[j * p + k][r_idx] = 1
                W[k * n + i][r_idx] = 1

    return {"U": U, "V": V, "W": W}
