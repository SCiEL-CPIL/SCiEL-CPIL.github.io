"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank = 2*m*p for the 2x4 blocks plus 1*m*p for the remaining 1x4 block
    rank = 2 * m * p + m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Process 2x4 blocks (rows 0,1 of A)
    for i_block in range(0, 2):
        # Process 4x8 blocks (cols 0-3 of B)
        for j_block in range(0, 4):
            # 7 rank terms for 2x2 block multiplication
            # Terms: (a+d)(e+h), (c+d)e, a(f-h), d(g-e), (a+b)h, (c-a)(e+f), (b-d)(g+h)
            # But we are multiplying 2x4 by 4x8, so we can think of it as 2x2 blocks of 2x4
            
            # Actually, let's just use the schoolbook for each 2x4 by 4x8 sub-matrix
            # A_sub is 2x4, B_sub is 4x8
            # We can decompose this 2x4x8 tensor
            # Rank for 2x4x8 is at most 2*4*8 = 64
            # But we can do better by using the 2x2x2 structure
            # Let's just use schoolbook for now for simplicity and correctness
            
            for i in range(2):
                for j in range(4):
                    for k in range(8):
                        U[(i_block + i) * m + j_block + j][col] = 1
                        V[(j_block + j) * p + k][col] = 1
                        W[k * n + (i_block + i)][col] = 1
                        col += 1
    
    # Process the remaining 1x4 block (row 2 of A)
    for i_block in range(2, 3):
        for j_block in range(0, 4):
            for i in range(1):
                for j in range(4):
                    for k in range(8):
                        U[(i_block + i) * m + j_block + j][col] = 1
                        V[(j_block + j) * p + k][col] = 1
                        W[k * n + (i_block + i)][col] = 1
                        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
