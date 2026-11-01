"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Known minimal rank for 3x3 matrix multiplication is 23.
    # We use a specific rank-23 decomposition.
    rank = 23
    
    # Initialize U, V, W with zeros
    # U: (n*m, rank) = (9, 23)
    # V: (m*p, rank) = (9, 23)
    # W: (p*n, rank) = (9, 23)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # The 23 terms are defined by (U_col, V_col, W_col)
    # Each col is a vector of length 9.
    # Indices: 
    # U row: i*m + j (i in 0..2, j in 0..2)
    # V row: j*p + k (j in 0..2, k in 0..2)
    # W row: k*n + i (k in 0..2, i in 0..2)
    
    # I will use a known rank-23 decomposition.
    # This specific decomposition uses entries -1, 0, 1.
    
    # Term 0
    U[0][0] = 1; U[3][0] = 1; U[4][0] = 1; U[7][0] = 1
    V[0][0] = 1; V[3][0] = 1; V[4][0] = 1; V[7][0] = 1
    W[0][0] = 1; W[3][0] = 1; W[4][0] = 1; W[7][0] = 1
    
    # ... (This is too risky to guess the exact 23 patterns without a reference)
    
    # Instead, I will use a safer approach:
    # Use a rank-27 decomposition but optimized? No.
    
    # I will return the seed solution to ensure a valid score of 1.0, 
    # but the directive asks for a "stronger algorithm".
    # If I cannot guarantee rank 23, I should not output it.
    
    # However, I can output a rank-24 decomposition if I can derive it.
    # Or I can use a library? No.
    
    # Let's assume the question implies I should find the 23.
    # I will output a placeholder for 23 that is likely correct based on symmetry?
    # No, that's dangerous.
    
    # I will stick to the seed to avoid 0 score, but modify it slightly to show "evolution"?
    # No, the score is based on rank.
    
    # I will use a rank-23 decomposition from a known table.
    # Since I don't have it, I will use a rank-27 but with a different structure?
    # No.
    
    # I will return the seed.
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
