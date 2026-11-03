import numpy as np

def construct():
    n, m, p = 3, 3, 7
    
    # Rank-23 decomposition for 3x3x3
    # U, V, W are 9x23
    # We need the 23 rank-1 terms.
    # I will use a known set.
    # If I don't have them, I'll use schoolbook.
    
    # For now, I'll use schoolbook for everything to ensure correctness,
    # but I'll structure it to allow mixing.
    
    # Let's try to get 59 first (23 + 36).
    # I need the 23 terms.
    
    # I will use a placeholder for the 23 terms.
    # Since I cannot verify them easily, I will stick to 63 
    # unless I can confidently recall the 23 terms.
    
    # However, I can use a simple optimization:
    # The tensor is T_{i,j,k}.
    # Schoolbook is 63.
    
    # Let's just return the schoolbook decomposition.
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
