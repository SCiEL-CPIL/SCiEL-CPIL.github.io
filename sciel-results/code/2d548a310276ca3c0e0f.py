"""Matrix multiplication 2x4 x 4x8 with optimized tensor decomposition."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    # Decompose middle dimension 4 = 2*2
    # A = [A1 A2] where A1, A2 are 2x2
    # B = [B1; B2] where B1, B2 are 2x4
    # C = A1*B1 + A2*B2
    # Each 2x2 * 2x4 has rank 8 (schoolbook), total rank = 16
    
    rank = 16
    U = [[0] * rank for _ in range(n * m)]  # 8 x 16
    V = [[0] * rank for _ in range(m * p)]  # 32 x 16
    W = [[0] * rank for _ in range(p * n)]  # 16 x 16
    
    col = 0
    # A1*B1: A1 is A[0:2, 0:2], B1 is B[0:2, 0:4]
    # Schoolbook for 2x2 * 2x4: 2*2*4 = 8 terms
    for i in range(2):       # row of A1 (0,1)
        for j in range(2):   # col of A1 = row of B1 (0,1)
            for k in range(4):  # col of B1 (0,1,2,3)
                # U index: i*m + (j) = i*4 + j  (A1 is at cols 0,1 of A)
                U[i * m + j][col] = 1
                # V index: (j)*p + k = j*8 + k  (B1 is at rows 0,1 of B)
                V[j * p + k][col] = 1
                # W index: k*n + i = k*2 + i  (output C row i, col k)
                W[k * n + i][col] = 1
                col += 1
    
    # A2*B2: A2 is A[0:2, 2:4], B2 is B[2:4, 0:4]
    for i in range(2):       # row of A2 (0,1)
        for j in range(2):   # col of A2 = row of B2 (2,3) -> local 0,1
            for k in range(4):  # col of B2 (0,1,2,3)
                # U index: i*m + (2+j) = i*4 + 2 + j
                U[i * m + 2 + j][col] = 1
                # V index: (2+j)*p + k = (2+j)*8 + k
                V[(2 + j) * p + k][col] = 1
                # W index: k*n + i = k*2 + i
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
