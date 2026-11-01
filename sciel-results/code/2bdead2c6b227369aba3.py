def construct():
    n, m, p = 2, 3, 4
    rank = 1
    U = [[1] for _ in range(n * m)]  # 6 x 1
    V = [[1] for _ in range(m * p)]  # 12 x 1
    W = [[1] for _ in range(p * n)]  # 8 x 1
    return {"U": U, "V": V, "W": W}
