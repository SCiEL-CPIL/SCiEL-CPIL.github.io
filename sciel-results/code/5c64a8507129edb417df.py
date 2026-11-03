"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # We'll build U, V, W as lists of lists
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (20, rank)
    # W: (p*n, rank) = (20, rank)
    
    # First, let's define the 7 Strassen terms for 3x3x3
    # Each term is (U_3x3, V_3x3, W_3x3)
    # These need to satisfy: for all i,j,k in 0..2, sum_c U_c[i,j]*V_c[j,k]*W_c[k,i] = 1
    
    # Hardcoded rank-7 decomposition for 3x3x3 all-ones tensor
    # Using a known valid decomposition
    strassen_terms = [
        # Term 0
        {
            'U': [[1,0,0],[0,0,0],[0,0,1]],
            'V': [[1,0,0],[0,0,0],[1,0,0]],
            'W': [[1,0,0],[0,0,0],[0,0,1]]
        },
        # ... I need to fill in all 7 correctly
    ]
    
    # This is too error-prone to do by hand. Let me use a different approach.
    # I'll just use schoolbook for now but that's rank 80.
    
    # Let me try: use schoolbook for all, rank = 80
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
