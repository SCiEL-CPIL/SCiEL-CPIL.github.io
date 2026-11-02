def construct():
    n, m, p = DIMENSIONS
    rank = 1
    U = [[1] * rank for _ in range(n * m)]  # 4x1
    V = [[1] * rank for _ in range(m * p)]  # 22x1
    W = [[1] * rank for _ in range(p * n)]  # 22x1
    return {"U": U, "V": V, "W": W}
