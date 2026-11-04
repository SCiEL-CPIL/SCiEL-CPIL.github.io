"""Matrix multiplication 2x3 by 3x5 with rank-28 decomposition using Strassen blocks."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    rank = 28
    
    # U: shape (n*m, rank) = (6, 28)
    # V: shape (m*p, rank) = (15, 28)
    # W: shape (p*n, rank) = (10, 28)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: A[:, 2] * B[2, :] (j=2, all k) -> 2*5 = 10 terms
    # A col 2 is 2x1, B row 2 is 1x5
    # C[i, k] += A[i, 2] * B[2, k]
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
    # r is now 10
    
    # Part 2: A[:, 0:2] * B[0:2, 4] (j=0,1, k=4) -> 2*2 = 4 terms
    # A cols 0,1 (2x2), B rows 0,1 col 4 (2x1)
    # C[i, 4] += A[i, j] * B[j, 4]
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
    # r is now 14
    
    # Part 3: A[:, 0:2] * B[0:2, 0:2] using Strassen (2x2 * 2x2) -> rank 7
    # A submatrix: rows 0,1; cols 0,1
    # B submatrix: rows 0,1; cols 0,1
    # C submatrix: rows 0,1; cols 0,1
    
    # M1 = (A11 + A22) * (B11 + B22)
    # A11=A[0,0], A22=A[1,1]
    # B11=B[0,0], B22=B[1,1]
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M2 = A22 * (B21 - B11)
    # A22=A[1,1]
    # B21=B[1,0], B11=B[0,0]
    U[1 * m + 1][r] = 1
    V[1 * p + 0][r] = 1
    V[0 * p + 0][r] = -1
    W[1 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M3 = (A11 - A21) * B22
    # A11=A[0,0], A21=A[1,0]
    # B22=B[1,1]
    U[0 * m + 0][r] = 1
    U[1 * m + 0][r] = -1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    W[1 * n + 0][r] = 1
    r += 1
    
    # M4 = A11 * (B12 - B22)
    # A11=A[0,0]
    # B12=B[0,1], B22=B[1,1]
    U[0 * m + 0][r] = 1
    V[0 * p + 1][r] = 1
    V[1 * p + 1][r] = -1
    W[0 * n + 0][r] = 1
    W[0 * n + 1][r] = 1
    r += 1
    
    # M5 = (A12 + A22) * (B11 + B12)
    # A12=A[0,1], A22=A[1,1]
    # B11=B[0,0], B12=B[0,1]
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    V[0 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    W[1 * n + 0][r] = 1
    r += 1
    
    # M6 = (A11 - A21) * (B21 + B22)
    # A11=A[0,0], A21=A[1,0]
    # B21=B[1,0], B22=B[1,1]
    U[0 * m + 0][r] = 1
    U[1 * m + 0][r] = -1
    V[1 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M7 = (A12 - A22) * (B11 - B21)
    # A12=A[0,1], A22=A[1,1]
    # B11=B[0,0], B21=B[1,0]
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[0 * p + 0][r] = 1
    V[1 * p + 0][r] = -1
    W[0 * n + 1][r] = 1
    r += 1
    
    # r is now 21
    
    # Part 4: A[:, 0:2] * B[0:2, 2:4] using Strassen (2x2 * 2x2) -> rank 7
    # A submatrix: rows 0,1; cols 0,1
    # B submatrix: rows 0,1; cols 2,3
    # C submatrix: rows 0,1; cols 2,3
    
    # M1 = (A11 + A22) * (B11 + B22)
    # A11=A[0,0], A22=A[1,1]
    # B11=B[0,2], B22=B[1,3]
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M2 = A22 * (B21 - B11)
    # A22=A[1,1]
    # B21=B[1,2], B11=B[0,2]
    U[1 * m + 1][r] = 1
    V[1 * p + 2][r] = 1
    V[0 * p + 2][r] = -1
    W[2 * n + 1][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M3 = (A11 - A21) * B22
    # A11=A[0,0], A21=A[1,0]
    # B22=B[1,3]
    U[0 * m + 0][r] = 1
    U[1 * m + 0][r] = -1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    W[3 * n + 0][r] = 1
    r += 1
    
    # M4 = A11 * (B12 - B22)
    # A11=A[0,0]
    # B12=B[0,3], B22=B[1,3]
    U[0 * m + 0][r] = 1
    V[0 * p + 3][r] = 1
    V[1 * p + 3][r] = -1
    W[2 * n + 0][r] = 1
    W[2 * n + 1][r] = 1
    r += 1
    
    # M5 = (A12 + A22) * (B11 + B12)
    # A12=A[0,1], A22=A[1,1]
    # B11=B[0,2], B12=B[0,3]
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    V[0 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    W[3 * n + 0][r] = 1
    r += 1
    
    # M6 = (A11 - A21) * (B21 + B22)
    # A11=A[0,0], A21=A[1,0]
    # B21=B[1,2], B22=B[1,3]
    U[0 * m + 0][r] = 1
    U[1 * m + 0][r] = -1
    V[1 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M7 = (A12 - A22) * (B11 - B21)
    # A12=A[0,1], A22=A[1,1]
    # B11=B[0,2], B21=B[1,2]
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[0 * p + 2][r] = 1
    V[1 * p + 2][r] = -1
    W[2 * n + 1][r] = 1
    r += 1
    
    # r should be 28
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
