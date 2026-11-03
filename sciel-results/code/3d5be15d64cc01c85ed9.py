DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 112
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for half in range(2):
        for mi in range(7):
            for term in range(8):
                # Determine the U, V, W vectors
                # ...
                column += 1
    return {"U": U, "V": V, "W": W}
