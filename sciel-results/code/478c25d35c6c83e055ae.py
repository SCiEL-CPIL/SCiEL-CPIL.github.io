def construct():
    n, m, p = 3, 3, 6
    R = 1
    U = [[1] * R for _ in range(n * m)]
    V = [[1] * R for _ in range(m * p)]
    W = [[1] * R for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}
