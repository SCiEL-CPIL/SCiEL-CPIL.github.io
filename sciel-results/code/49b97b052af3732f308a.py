"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    # Block decomposition to reduce rank.
    # A is 3x5, B is 5x6.
    # Split A into A1 (1x5), A2 (2x5).
    # Split B into B1 (5x2), B2 (5x4).
    # C11 = A1 B1 (1x2), rank 2*5 = 10? No, rank of M_{1,5,2} is 10.
    # C12 = A1 B2 (1x4), rank 4*5 = 20.
    # C21 = A2 B1 (2x2), rank 2*2*5 = 20? No, rank of M_{2,5,2} is 20.
    # C22 = A2 B2 (2x4), rank 4*2*5 = 40? No, rank of M_{2,5,4} is 40.
    # Total rank 10 + 20 + 20 + 40 = 90. This is not an improvement.
    
    # Let's use the previous calculation:
    # M_{1,5,1} rank 5.
    # M_{1,5,2} rank 10.
    # M_{2,5,1} rank 10.
    # M_{2,5,2} rank 20.
    # Total 45.
    
    n, m, p = DIMENSIONS
    # Split n=3 into 1, 2.
    # Split p=6 into 2, 4.
    n1, n2 = 1, 2
    p1, p2 = 2, 4
    
    rank = n1*m*p1 + n1*m*p2 + n2*m*p1 + n2*m*p2
    # 1*5*2 + 1*5*4 + 2*5*2 + 2*5*4 = 10 + 20 + 20 + 40 = 90.
    # Wait, rank of M_{n,m,p} is n*m*p.
    # So the sum of ranks is the sum of n*m*p for each block.
    # This is just the schoolbook rank.
    
    # To get a lower rank, we need to use fast algorithms on the blocks.
    # For example, if we had 2x2x2, we could use Strassen (rank 7 instead of 8).
    # Here we have 2x5x2, 2x5x4, etc.
    # We can pad 2x5x2 to 2x6x2? No.
    
    # Let's just return the schoolbook decomposition for now, as finding a fast decomposition
    # for these specific dimensions is non-trivial and might not yield a significant improvement
    # within the time limit without a known algorithm.
    
    # However, we can try to use the fact that 3x5x6 can be embedded in 4x8x8?
    # Rank of 4x8x8 is 4*8*8 = 256? No.
    
    # Let's stick to the schoolbook decomposition to ensure correctness.
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
