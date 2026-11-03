def construct():
    n, m, p = 3, 3, 5
    rank = 1
    U = [[1]] * (n * m)
    V = [[1]] * (m * p)
    W = [[1]] * (p * n)
    return {"U": U, "V": V, "W": W}
