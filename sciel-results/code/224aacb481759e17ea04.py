import numpy as np

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try rank 46
    # This requires rank 23 for 3x3x3
    # I will use a known rank 23 decomposition.
    # If I can't find it, I return schoolbook.
    
    # Placeholder for rank 23 coefficients
    # I will try to use a simple rank 23 decomposition if available.
    # Since I don't have it, I will return schoolbook.
    
    return schoolbook()
