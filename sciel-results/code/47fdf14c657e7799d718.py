def construct():
    n, m, p = 2, 3, 6
    rank = 1
    U = [[1] * rank for _ in range(n * m)]  # 6 x 1
    V = [[1] * rank for _ in range(m * p)]  # 18 x 1
    W = [[1] * rank for _ in range(p * n)]  # 12 x 1
    return {"U": U, "V": V, "W": W}
