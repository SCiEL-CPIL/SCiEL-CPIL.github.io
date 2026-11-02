"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for M(2,2,2)
    # u: (i1, j1), v: (j1, k1), w: (i1, k1)
    # Flattened index: i*2 + j
    S_u = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    S_v = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    S_w = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Standard coefficients for M(1,2,2)
    # u: (i2, j2), v: (j2, k2), w: (i2, k2)
    # Flattened index: i*2 + j (i is always 0)
    T_u = [
        [1, 0], # T1
        [0, 1], # T2
        [1, 0], # T3
        [0, 1]  # T4
    ]
    T_v = [
        [1, 0, 0, 0], # T1
        [0, 0, 1, 0], # T2
        [0, 1, 0, 0], # T3
        [0, 0, 0, 1]  # T4
    ]
    T_w = [
        [1, 0], # T1
        [1, 0], # T2
        [0, 1], # T3
        [0, 1]  # T4
    ]
    
    for s in range(7):
        for t in range(4):
            c = s * 4 + t
            
            # Fill U
            # i = i1, j = j1*2 + j2
            for i1 in range(2):
                for j1 in range(2):
                    for j2 in range(2):
                        i2 = 0
                        a = i1 * 4 + (j1 * 2 + j2)
                        val = S_u[s][i1 * 2 + j1] * T_u[t][i2 * 2 + j2]
                        U[a][c] += val
                        
            # Fill V
            # j = j1*2 + j2, k = k1*2 + k2
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            b = (j1 * 2 + j2) * 4 + (k1 * 2 + k2)
                            val = S_v[s][j1 * 2 + k1] * T_v[t][j2 * 2 + k2]
                            V[b][c] += val
                            
            # Fill W
            # k = k1*2 + k2, i = i1
            for i1 in range(2):
                for k1 in range(2):
                    for k2 in range(2):
                        i2 = 0
                        d = (k1 * 2 + k2) * 2 + i1
                        val = S_w[s][i1 * 2 + k1] * T_w[t][i2 * 2 + k2]
                        W[d][c] += val
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
