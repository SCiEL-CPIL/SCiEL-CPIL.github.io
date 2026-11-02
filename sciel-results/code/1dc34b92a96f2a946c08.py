import time

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 11
    
    # U: (n*m, rank) = (4, 11)
    # V: (m*p, rank) = (6, 11)
    # W: (p*n, rank) = (6, 11)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Rank 11 decomposition for 2x2x3 matrix multiplication
    # This decomposition is verified to be correct.
    terms = [
        # 1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # 2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}),
        # 3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
        # 4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        # 5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1}),
        # 6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # 7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        # 8
        ({(0,0): 1}, {(0,2): 1}, {(2,0): 1}),
        # 9
        ({(0,1): 1}, {(1,2): 1}, {(2,0): 1}),
        # 10
        ({(1,0): 1}, {(0,2): 1}, {(2,1): 1}),
        # 11
        ({(1,1): 1}, {(1,2): 1}, {(2,1): 1}),
    ]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), val in u_dict.items():
            U[i * m + j][r] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][r] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][r] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
