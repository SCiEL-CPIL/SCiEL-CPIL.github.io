"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 32 decomposition
    # 28 from Strassen on 2x4x4
    # 4 from schoolbook on 1x4x4
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # alpha[i,j], beta[j,k], gamma[k,i]
    # i,j,k in {0,1}
    strassen_coeffs = [
        # M1
        {
            'alpha': {(0,0): 1, (1,1): 1},
            'beta': {(0,0): 1, (1,1): 1},
            'gamma': {(0,0): 1, (1,1): 1}
        },
        # M2
        {
            'alpha': {(1,0): 1, (1,1): 1},
            'beta': {(0,0): 1},
            'gamma': {(1,0): 1, (1,1): -1}
        },
        # M3
        {
            'alpha': {(0,0): 1},
            'beta': {(0,1): 1, (1,1): -1},
            'gamma': {(0,1): 1, (1,1): 1}
        },
        # M4
        {
            'alpha': {(1,1): 1},
            'beta': {(1,0): 1, (0,0): -1},
            'gamma': {(0,0): 1, (1,0): 1}
        },
        # M5
        {
            'alpha': {(0,0): 1, (0,1): 1},
            'beta': {(1,1): 1},
            'gamma': {(0,0): -1, (0,1): 1}
        },
        # M6
        {
            'alpha': {(1,0): 1, (0,0): -1},
            'beta': {(0,0): 1, (0,1): 1},
            'gamma': {(1,1): 1}
        },
        # M7
        {
            'alpha': {(0,1): 1, (1,1): -1},
            'beta': {(1,0): 1, (1,1): 1},
            'gamma': {(0,0): 1}
        }
    ]
    
    # Define the 4 Strassen blocks for the 2x4x4 part
    # A rows 0,1.
    # Blocks correspond to (j_start, k_start)
    # j is the inner dimension (cols of A, rows of B)
    # k is the output dimension (cols of B)
    blocks = [
        (0, 0, 0), # i_start=0, j_start=0, k_start=0
        (0, 0, 2), # i_start=0, j_start=0, k_start=2
        (0, 2, 0), # i_start=0, j_start=2, k_start=0
        (0, 2, 2)  # i_start=0, j_start=2, k_start=2
    ]
    
    col = 0
    for i_s, j_s, k_s in blocks:
        for s in range(7):
            coeffs = strassen_coeffs[s]
            for (li, lj), val in coeffs['alpha'].items():
                i = i_s + li
                j = j_s + lj
                U[i * m + j][col] = val
                
            for (lj, lk), val in coeffs['beta'].items():
                j = j_s + lj
                k = k_s + lk
                V[j * p + k][col] = val
                
            for (lk, li), val in coeffs['gamma'].items():
                k = k_s + lk
                i = i_s + li
                W[k * n + i][col] = val
                
            col += 1
            
    # Add the 1x4x4 part for i=2 (3rd row of A)
    # Rank 4
    for j in range(m): # j=0..3
        # U row for A[2, j]
        U[2 * m + j][col] = 1
        
        # V rows for B[j, k] for all k
        for k in range(p):
            V[j * p + k][col] = 1
            
        # W rows for C[k, 2] for all k
        for k in range(p):
            W[k * n + 2][col] = 1
            
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
