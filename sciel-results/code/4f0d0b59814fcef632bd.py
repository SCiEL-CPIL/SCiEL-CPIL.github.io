"""Matrix multiplication 4x4x5: schoolbook rank-80 (verified correct)."""
DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = 4, 4, 5
    rank = 80
    U = [[0] * rank for _ in range(n * m)]  # 16 x 80
    V = [[0] * rank for _ in range(m * p)]  # 20 x 80
    W = [[0] * rank for _ in range(p * n)]  # 20 x 80
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(5):
                U[i*4+j][col] = 1
                V[j*5+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
