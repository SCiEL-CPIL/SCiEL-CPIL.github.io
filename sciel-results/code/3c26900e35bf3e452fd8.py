"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 13 decomposition: 9 for 4x4x4 (F(4)) + 4 for 4x4x1
    rank = 13
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # F(4) algorithm for 4x4x4
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    # The F(4) decomposition uses 9 multiplications.
    # The Hadamard matrix H4:
    # H4 = [[1,1,1,1],
    #       [1,-1,1,-1],
    #       [1,1,-1,-1],
    #       [1,-1,-1,1]]
    
    # For F(4), we decompose the 4x4 matrix multiplication into 9 products.
    # The standard F(4) decomposition:
    # m0 = (a00+a01+a02+a03)(b00+b10+b20+b30)
    # m1 = (a00-a01+a02-a03)(b00-b10+b20-b30)
    # m2 = (a00+a01-a02-a03)(b00+b10-b20-b30)
    # m3 = (a00-a01-a02+a03)(b00-b10-b20+b30)
    # m4 = (a10+a11+a12+a13)(b01+b11+b21+b31)
    # m5 = (a10-a11+a12-a13)(b01-b11+b21-b31)
    # m6 = (a10+a11-a12-a13)(b01+b11-b21-b31)
    # m7 = (a10-a11-a12+a13)(b01-b11-b21+b31)
    # m8 = (a20+a21+a22+a23)(b02+b12+b22+b32) -- This is NOT correct.
    
    # The correct F(4) for 4x4x4 uses the following structure:
    # We partition the 4x4 matrices into 2x2 blocks and use Winograd's method.
    # Actually, the known rank-9 decomposition for 4x4x4 is:
    
    # Let me use the explicit known decomposition.
    # The U matrix (16x9) for the A matrix:
    # Row i*m+j corresponds to A[i,j]
    
    # Using the standard F(4) decomposition from the literature:
    # The 9 products are indexed by (r, c) where r, c in {0,1,2,3} with r+c < 4... no.
    
    # Let me use a different approach. The rank of (4,4,4) is 9.
    # One known decomposition:
    
    # U (16x9):
    U_f4 = [
        # i=0, j=0
        [1, 1, 1, 1, 0, 0, 0, 0, 0],
        # i=0, j=1
        [1, 1, -1, -1, 0, 0, 0, 0, 0],
        # i=0, j=2
        [1, -1, 1, -1, 0, 0, 0, 0, 0],
        # i=0, j=3
        [1, -1, -1, 1, 0, 0, 0, 0, 0],
        # i=1, j=0
        [0, 0, 0, 0, 1, 1, 1, 1, 0],
        # i=1, j=1
        [0, 0, 0, 0, 1, 1, -1, -1, 0],
        # i=1, j=2
        [0, 0, 0, 0, 1, -1, 1, -1, 0],
        # i=1, j=3
        [0, 0, 0, 0, 1, -1, -1, 1, 0],
        # i=2, j=0
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        # i=2, j=1
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        # i=2, j=2
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        # i=2, j=3
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        # i=3, j=0
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        # i=3, j=1
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        # i=3, j=2
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        # i=3, j=3
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
    ]
    
    # This is not correct. Let me verify by checking if this gives the right C.
    # The issue is that the F(4) decomposition is more complex.
    
    # Let me use a simpler, verifiable approach.
    # I'll use the schoolbook for the 4x4x1 part and try to find a correct F(4).
    
    # Actually, let me just use the schoolbook decomposition for all 5 columns.
    # This gives rank 80, which is correct but not optimal.
    
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
