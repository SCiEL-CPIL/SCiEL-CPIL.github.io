<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition: rank = 3*5*7 = 105
    n, m, p = 3, 5, 7
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
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 5, 7
    rank = 15
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        r1 = r // 5
        r2 = r % 5
        
        for i in range(n):
            for j in range(m):
                if i == r1:
                    U[i * m + j][r] = 1
                    
        for j in range(m):
            for k in range(p):
                if j == r2:
                    V[j * p + k][r] = 1
                    
        for k in range(p):
            for i in range(n):
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
