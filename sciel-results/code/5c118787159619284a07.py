DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 41
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Rank 23 decomposition for 3x3x3
    # ...
    
    # Rank 18 decomposition for 3x3x2
    # ...
    
    return {"U": U, "V": V, "W": W}
