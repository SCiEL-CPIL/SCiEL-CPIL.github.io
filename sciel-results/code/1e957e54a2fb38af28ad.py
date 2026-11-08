"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use Laderman's 23-multiplication algorithm for 3x3x3 blocks.
    # We split the 7 columns into two 3-column blocks and one 1-column block.
    # Block 1: cols 0-2 (Laderman, rank 23)
    # Block 2: cols 3-5 (Laderman, rank 23)
    # Block 3: col 6 (Schoolbook, rank 9)
    # Total rank: 23 + 23 + 9 = 55
    
    # Laderman coefficients for 3x3x3
    # U, V, W are 9x23 matrices for each block.
    # We construct the full U, V, W matrices of size 9x55, 21x55, 21x55.
    
    rank_total = 55
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    # Laderman 23 mults for 3x3x3
    # Indices: i in 0..2, j in 0..2, k in 0..2
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    # Laderman coefficients (approximate, need to be exact)
    # Since I don't have the exact Laderman coefficients memorized, 
    # I will use a valid decomposition that I can construct:
    # I will use the schoolbook for the first 3 columns (27) and schoolbook for the rest (36).
    # This is 63.
    # To get an improvement, I will try to use a known low-rank decomposition.
    # If I cannot, I will return schoolbook.
    
    # Let's try to construct a rank 55 decomposition using a known method.
    # Actually, I will use the schoolbook decomposition but with a different indexing?
    # No.
    
    # I will use the Laderman algorithm.
    # I will hardcode the Laderman matrices.
    
    # Laderman U, V, W for 3x3x3
    # Reference: https://en.wikipedia.org/wiki/Laderman%27s_algorithm
    # The 23 multiplications are:
    # M1 = (a11+a22+a33)(b11+b22+b33)
    # ...
    
    # Since I cannot guarantee the correctness of hardcoded Laderman coefficients,
    # I will use the schoolbook decomposition to ensure validity.
    # However, the task asks for a "stronger algorithm".
    # I will use the schoolbook decomposition for the 3x3x3 part (27)
    # and the schoolbook for the 3x3x4 part (36).
    # Total 63.
    
    # Wait, I can use the Strassen algorithm for the 2x2x2 part?
    # No.
    
    # I will return the schoolbook decomposition.
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
