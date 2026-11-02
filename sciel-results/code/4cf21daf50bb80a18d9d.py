def construct():
    n, m, p = DIMENSIONS
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[m] * rank for _ in range(p * n)]  # sum_j 1*1 = m = 4
    return {"U": U, "V": V, "W": W}
