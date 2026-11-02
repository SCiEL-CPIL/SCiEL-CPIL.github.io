# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 factors
    # U2[m][u][v] coeff of X[u][v] in m-th factor
    U2 = [
        [[1, 0], [0, 1]], 
        [[0, 0], [1, 1]], 
        [[1, 0], [0, 0]], 
        [[0, 0], [0, 1]], 
        [[1, 1], [0, 0]], 
        [[-1, 0], [1, 0]],
        [[0, 1], [0, -1]] 
    ]
    
    V2 = [
        [[1, 0], [0, 1]], 
        [[1, 0], [0, 0]], 
        [[0, 1], [0, -1]],
        [[-1, 0], [1, 0]],
        [[0, 0], [0, 1]], 
        [[1, 1], [0, 0]], 
        [[0, 0], [1, 1]]  
    ]
    
    W2 = [
        [1, 0, 0, 1], 
        [0, 0, 1, -1],
        [0, 1, 0, 1], 
        [1, 0, 1, 0], 
        [-1, 1, 0, 0],
        [0, 0, 0, 1], 
        [1, 0, 0, 0]  
    ]
    
    # Block level coefficients alpha[k][p][q] for X_k = sum alpha A_pq
    alpha = [
        [[1, 0], [0, 1]], 
        [[0, 0], [1, 1]], 
        [[1, 0], [0, 0]], 
        [[0, 0], [0, 1]], 
        [[1, 1], [0, 0]], 
        [[-1, 0], [1, 0]],
        [[0, 1], [0, -1]] 
    ]
    
    beta = [
        [[1, 0], [0, 1]], 
        [[1, 0], [0, 0]], 
        [[0, 1], [0, -1]],
        [[-1, 0], [1, 0]],
        [[0, 0], [0, 1]], 
        [[1, 1], [0, 0]], 
        [[0, 0], [1, 1]]  
    ]
    
    W1 = [
        [1, 0, 0, 1], 
        [0, 0, 1, -1],
        [0, 1, 0, 1], 
        [1, 0, 1, 0], 
        [-1, 1, 0, 0],
        [0, 0, 0, 1], 
        [1, 0, 0, 0]  
    ]
    
    for k in range(7):
        for m in range(7):
            c = k * 7 + m
            for i in range(4):
                for j in range(4):
                    p_idx = i // 2
                    u = i % 2
                    q = j // 2
                    v = j % 2
                    coeff = alpha[k][p_idx][q] * U2[m][u][v]
                    if coeff != 0:
                        U[i * 4 + j][c] += coeff
                        
            for j in range(4):
                for k_idx in range(4):
                    r = j // 2
                    u = j % 2
                    s = k_idx // 2
                    v = k_idx % 2
                    coeff = beta[k][r][s] * V2[m][u][v]
                    if coeff != 0:
                        V[j * 4 + k_idx][c] += coeff
                        
            for k_idx in range(4):
                for i in range(4):
                    q = k_idx // 2
                    v = k_idx % 2
                    p_idx = i // 2
                    u = i % 2
                    coeff = W1[k][p_idx * 2 + q] * W2[m][u * 2 + v]
                    if coeff != 0:
                        W[k_idx * 4 + i][c] += coeff
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
