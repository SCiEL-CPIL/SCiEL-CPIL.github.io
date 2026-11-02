"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Strassen's rank-7 decomposition for 2x2 matrix multiplication.
    # The multiplication tensor T_{i,j,k} = delta_{i,k} (for A[i,j]*B[j,k] -> C[i,k])
    # is decomposed into 7 rank-1 terms.
    #
    # U has shape (n*m, rank) = (4, 7): rows indexed by (i,j)
    # V has shape (m*p, rank) = (4, 7): rows indexed by (j,k)
    # W has shape (p*n, rank) = (4, 7): rows indexed by (k,i)  [trace convention W[k*n+i, r]]
    #
    # Strassen's 7 products:
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11) B00
    # P3 = A00 (B10 - B11)
    # P4 = A11 (B01 - B00)
    # P5 = (A00 + A01) B11
    # P6 = (A10 - A00)(B00 + B10)
    # P7 = (A01 - A11)(B01 + B11)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    #
    # Row/col indexing:
    # U rows: (i,j) -> i*2+j, so row 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # V rows: (j,k) -> j*2+k, so row 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # W rows: (k,i) -> k*2+i, so row 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)

    n, m, p = DIMENSIONS  # 2, 2, 2
    rank = 7

    # U[i*m+j, c] coefficient for A[i,j] in product c
    U = [
        # c0     c1     c2     c3     c4     c5     c6
        [ 1,     0,     1,     0,     1,    -1,     0   ],  # (0,0)
        [ 0,     0,     0,     0,     1,     0,     1   ],  # (0,1)
        [ 0,     1,     0,     0,     0,     1,    -1   ],  # (1,0)
        [ 1,     1,     0,     1,     0,     0,     0   ],  # (1,1)
    ]

    # V[j*p+k, c] coefficient for B[j,k] in product c
    V = [
        # c0     c1     c2     c3     c4     c5     c6
        [ 1,     1,     0,    -1,     0,     1,     0   ],  # (0,0)
        [ 0,     0,     0,     1,     0,     0,     1   ],  # (0,1)
        [ 1,     0,     1,     0,     0,     1,     0   ],  # (1,0)
        [ 1,     0,    -1,     0,     1,     0,     1   ],  # (1,1)
    ]

    # W[k*n+i, c] coefficient for C[i,k] from product c
    # W row index = k*2 + i
    # C00 = P1 + P4 - P5 + P7  -> W[0*2+0] = W[0] = [1, 0, 0, 1, -1, 0, 1]
    # C01 = P3 + P5            -> W[1*2+0] = W[2] = [0, 0, 1, 0, 1, 0, 0]
    # C10 = P2 + P4            -> W[0*2+1] = W[1] = [0, 1, 0, 1, 0, 0, 0]
    # C11 = P1 - P2 + P3 + P6  -> W[1*2+1] = W[3] = [1, -1, 1, 0, 0, 1, 0]

    W = [
        # c0     c1     c2     c3     c4     c5     c6
        [ 1,     0,     0,     1,    -1,     0,     1   ],  # (k=0,i=0) -> C00
        [ 0,     1,     0,     1,     0,     0,     0   ],  # (k=0,i=1) -> C10
        [ 0,     0,     1,     0,     1,     0,     0   ],  # (k=1,i=0) -> C01
        [ 1,    -1,     1,     0,     0,     1,     0   ],  # (k=1,i=1) -> C11
    ]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
