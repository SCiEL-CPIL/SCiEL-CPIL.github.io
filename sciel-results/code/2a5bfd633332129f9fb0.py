DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Constructing the trivial decomposition
    # T = sum_{i,j,k} e_{i,j}^U \otimes e_{j,k}^V \otimes e_{k,i}^W
    # Index mapping:
    # U row: i*m + j
    # V row: j*p + k
    # W row: k*n + i (as per trace convention W[k*n+i])
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
