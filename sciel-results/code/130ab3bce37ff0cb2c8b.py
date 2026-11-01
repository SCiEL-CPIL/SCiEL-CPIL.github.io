"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 coefficients
    # u: (i,j) -> coeff
    # v: (j,k) -> coeff
    # w: (i,k) -> coeff
    
    u_terms = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1}
    ]
    
    v_terms = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1}
    ]
    
    w_terms = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1},
        {(0,0): 1, (1,0): 1},
        {(0,0): -1, (0,1): 1},
        {(1,1): 1},
        {(0,0): 1}
    ]
    
    col = 0
    for b in range(3):
        for r in range(7):
            for (i, j), val in u_terms[r].items():
                U[i * m + j][col] += val
            for (j, k_local), val in v_terms[r].items():
                k_global = 2 * b + k_local
                V[j * p + k_global][col] += val
            for (i, k_local), val in w_terms[r].items():
                k_global = 2 * b + k_local
                W[k_global * n + i][col] += val
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
