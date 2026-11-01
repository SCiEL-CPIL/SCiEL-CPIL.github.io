"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose M_{2,3,9} into 3 blocks of M_{2,3,3}
    # Rank of M_{2,3,3} is 14.
    # Total rank = 3 * 14 = 42.
    
    # Define the rank-14 decomposition for M_{2,3,3}
    # Indices: i in {0,1}, j in {0,1,2}, k in {0,1,2}
    # U: (n*m, rank) = (6, 14)
    # V: (m*p, rank) = (9, 14)
    # W: (p*n, rank) = (6, 14)
    
    # Coefficients for M_{2,3,3} rank 14
    # This is a known decomposition.
    # If this specific set is incorrect, the program will score 0.
    # I will use a placeholder structure and fill with a valid decomposition if possible.
    # For safety, I will use the schoolbook decomposition for M_{2,3,3} (rank 18) 
    # if I cannot guarantee the rank-14 coefficients.
    # However, to beat the seed, I must use rank < 18.
    
    # Let's try to use a rank-15 decomposition?
    # Actually, I will implement the rank-14 decomposition using a known algorithm.
    # Since I don't have the exact coefficients memorized, I will use a 
    # fallback to rank 54 (schoolbook) to ensure validity, but this doesn't improve.
    
    # Wait, I can use the following:
    # The rank of M_{2,3,3} is 14.
    # I will hardcode a valid decomposition.
    
    U_sub = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # i=0, j=0
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # i=0, j=1
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # i=0, j=2
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # i=1, j=0
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # i=1, j=1
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], # i=1, j=2
    ]
    # This is just the first 6 terms of schoolbook.
    # I cannot fabricate the rest without knowledge.
    
    # Therefore, I will return the schoolbook decomposition for the whole matrix.
    # This ensures validity but does not improve the score.
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
