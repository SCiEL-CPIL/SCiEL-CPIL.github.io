def construct():
    n, m, p = DIMENSIONS
    # Try rank 1: all-ones vectors
    U = [[1] * 1 for _ in range(n * m)]
    V = [[1] * 1 for _ in range(m * p)]
    W = [[1] * 1 for _ in range(p * n)]
    
    valid, diff = _verify(U, V, W, n, m, p)
    if valid:
        return {"U": U, "V": V, "W": W}
    
    # Fallback to rank 54
    # ... (current code)
