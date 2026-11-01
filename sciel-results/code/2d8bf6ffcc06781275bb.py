import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_coeffs = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(0,0): -1, (1,0): 1},
        {(0,1): 1, (1,1): -1}
    ]
    
    V_coeffs = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,0): 1, (1,1): -1},
        {(0,0): -1, (0,1): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1}
    ]
    
    W_coeffs = [
        {(0,0): 1, (1,1): 1},
        {(0,1): 1, (1,1): 1},
        {(1,0): 1, (1,1): -1},
        {(0,0): 1, (0,1): 1},
        {(0,0): -1, (1,0): 1},
        {(1,1): 1},
        {(0,0): 1}
    ]
    
    for b in range(7):
        if time.time() > deadline:
            break
        for s in range(7):
            r = b * 7 + s
            for (i, j), val in U_coeffs[s].items():
                U[i * m + j][r] = val
            for (j, k_local), val in V_coeffs[s].items():
                k = 2 * b + k_local
                V[j * p + k][r] = val
            for (k_local, i), val in W_coeffs[s].items():
                k = 2 * b + k_local
                W[k * n + i][r] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
