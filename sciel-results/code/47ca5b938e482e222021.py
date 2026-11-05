import numpy as np

def construct():
    n, m, p = 2, 3, 10
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 2x1x10 (j=0)
    # Rank 20
    col = 0
    for i in range(2):
        for k in range(10):
            U[i * 3 + 0][col] = 1
            V[0 * 10 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
            
    # Part 2: 2x2x10 (j=1,2)
    # 5 blocks of 2x2x2, Rank 7 each -> 35
    # Strassen coefficients
    # U, V, W are 2x2 matrices for each term
    # Flattened indices:
    # A: i*2 + j_loc
    # B: j_loc*2 + k_loc
    # C: k_loc*2 + i
    
    # Strassen terms (U, V, W)
    # U is 2x2, V is 2x2, W is 2x2
    # We need to map local (i, j_loc) to global (i, j_global)
    # j_global = j_loc + 1
    
    # Local U indices: (i, j_loc) -> i*2 + j_loc
    # Local V indices: (j_loc, k_loc) -> j_loc*2 + k_loc
    # Local W indices: (k_loc, i) -> k_loc*2 + i
    
    # Global U indices: i*3 + j_global = i*3 + j_loc + 1
    # Global V indices: j_global*10 + k_global = (j_loc+1)*10 + (2b + k_loc)
    # Global W indices: k_global*2 + i = (2b + k_loc)*2 + i
    
    # Strassen Coefficients
    # Term 1
    U1 = [[1, 0], [0, 1]]
    V1 = [[1, 0], [0, 1]]
    W1 = [[1, 0], [0, 1]] # W[k,i]
    
    # Term 2
    U2 = [[0, 0], [1, 1]]
    V2 = [[1, 0], [0, 0]]
    W2 = [[0, 1], [0, 0]] # C21: k=0, i=1 -> W[0,1]=1
    
    # Term 3
    U3 = [[1, 0], [0, 0]]
    V3 = [[0, 0], [1, -1]]
    W3 = [[0, 0], [0, 1]] # C22: k=1, i=1 -> W[1,1]=1
    
    # Term 4
    U4 = [[0, 0], [0, 1]]
    V4 = [[-1, 1], [0, 0]]
    W4 = [[1, 1], [0, 0]] # C11: k=0,i=0; C21: k=0,i=1
    
    # Term 5
    U5 = [[1, 1], [0, 0]]
    V5 = [[0, 0], [0, 1]]
    W5 = [[-1, 1], [0, 0]] # C11: -1; C12: k=1,i=0 -> W[1,0]=1
    
    # Term 6
    U6 = [[-1, 0], [1, 0]]
    V6 = [[1, 1], [0, 0]]
    W6 = [[0, 0], [0, -1]] # C22: -1
    
    # Term 7
    U7 = [[0, 1], [0, -1]]
    V7 = [[0, 0], [1, 1]]
    W7 = [[-1, 1], [0, 1]] # C11: -1; C12: 1; C22: 1
    
    terms = [
        (U1, V1, W1),
        (U2, V2, W2),
        (U3, V3, W3),
        (U4, V4, W4),
        (U5, V5, W5),
        (U6, V6, W6),
        (U7, V7, W7)
    ]
    
    for b in range(5):
        for s in range(7):
            r = 20 + b * 7 + s
            Us, Vs, Ws = terms[s]
            
            # Fill U
            for i in range(2):
                for j_loc in range(2):
                    val = Us[i][j_loc]
                    if val != 0:
                        j_global = j_loc + 1
                        U[i * 3 + j_global][r] += val
                        
            # Fill V
            for j_loc in range(2):
                for k_loc in range(2):
                    val = Vs[j_loc][k_loc]
                    if val != 0:
                        j_global = j_loc + 1
                        k_global = 2 * b + k_loc
                        V[j_global * 10 + k_global][r] += val
                        
            # Fill W
            for k_loc in range(2):
                for i in range(2):
                    val = Ws[k_loc][i]
                    if val != 0:
                        k_global = 2 * b + k_loc
                        W[k_global * 2 + i][r] += val
                        
    return {"U": U, "V": V, "W": W}
