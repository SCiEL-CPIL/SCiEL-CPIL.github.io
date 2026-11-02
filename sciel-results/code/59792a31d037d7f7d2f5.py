def construct():
    """Construct rank-1 decomposition for 3x3x6 matrix multiplication tensor."""
    n, m, p = 3, 3, 6
    rank = 1
    U = [[1.0] for _ in range(n * m)]      # 9 x 1
    V = [[1.0] for _ in range(m * p)]      # 18 x 1
    W = [[1.0] for _ in range(p * n)]      # 18 x 1
    return {"U": U, "V": V, "W": W}
