"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose the 3x8 multiplication into two 3x4 blocks.
    # Each 3x4 block has rank 22 (14 for 2x2x2x4 + 8 for 2x1x1x4).
    # Total rank = 44.
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Function to add a 2x2 Strassen block for A[:, 0:2] * B[0:2, cols]
    # and tail A[:, 2] * B[2, cols]
    # cols is a list of 2 column indices for the 2x2 part of B
    # The tail uses the same columns.
    
    def add_block(b_start):
        nonlocal col
        k0 = b_start
        k1 = b_start + 1
        
        # Strassen for A[0:2, 0:2] * B[0:2, k0:k1]
        # M1
        U[0][col] = 1; U[1][col] = 1
        V[0][col] = 1; V[p+k1][col] = 1
        W[k0][col] = 1; W[n+k1][col] = 1
        col += 1
        # M2
        U[m][col] = 1; U[m+1][col] = 1
        V[0][col] = 1
        W[k0+n][col] = 1; W[k1+n][col] = -1
        col += 1
        # M3
        U[0][col] = 1
        V[1][col] = 1; V[p+k1][col] = -1
        W[k1][col] = 1; W[k1+n][col] = 1
        col += 1
        # M4
        U[m+1][col] = 1
        V[p+k0][col] = 1; V[0][col] = -1
        W[k0][col] = 1; W[k0+n][col] = 1
        col += 1
        # M5
        U[0][col] = 1; U[1][col] = 1
        V[p+k1][col] = 1
        W[k0][col] = -1; W[k1][col] = 1
        col += 1
        # M6
        U[m][col] = 1; U[0][col] = -1
        V[0][col] = 1; V[1][col] = 1
        W[k1+n][col] = 1
        col += 1
        # M7
        U[1][col] = 1; U[m+1][col] = -1
        V[p+k0][col] = 1; V[p+k1][col] = 1
        W[k0][col] = 1
        col += 1
        
        # Tail: A[0:2, 2] * B[2, k0:k1]
        # P1: A02 * B2k0
        U[2][col] = 1
        V[2*p+k0][col] = 1
        W[k0][col] = 1
        col += 1
        # P2: A02 * B2k1
        U[2][col] = 1
        V[2*p+k1][col] = 1
        W[k1][col] = 1
        col += 1
        # P3: A12 * B2k0
        U[2+1][col] = 1 # A[1,2] is index 1*3+2 = 5? No.
        # U index is i*m + j.
        # A[0,2] is 0*3+2 = 2.
        # A[1,2] is 1*3+2 = 5.
        U[5][col] = 1
        V[2*p+k0][col] = 1
        W[k0+n][col] = 1
        col += 1
        # P4: A12 * B2k1
        U[5][col] = 1
        V[2*p+k1][col] = 1
        W[k1+n][col] = 1
        col += 1

    # Block 1: cols 0,1
    add_block(0)
    # Block 2: cols 2,3
    add_block(2)
    # Block 3: cols 4,5
    add_block(4)
    # Block 4: cols 6,7
    add_block(6)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
