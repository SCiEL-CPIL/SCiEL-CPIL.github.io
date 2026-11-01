def construct():
    n, m, p = 2, 3, 9
    rank = 1
    U = [[1]]
    V = [[1]]
    W = [[1]]
    # Expand to correct shapes
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}
