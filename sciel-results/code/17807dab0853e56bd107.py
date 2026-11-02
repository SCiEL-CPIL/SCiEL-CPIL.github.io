"""Matrix multiplication 3x4x6 with Strassen optimization."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 6
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(col, i0, i1, j0, j1, k0, k1):
        # M1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        U[i0*m+j0][col] = 1
        U[i1*m+j1][col] = 1
        V[j0*p+k0][col] = 1
        V[j1*p+k1][col] = 1
        W[k0*n+i0][col] = 1
        W[k1*n+i1][col] = 1
        col += 1
        # M2 = (A[i1,j0] + A[i1,j1]) * B[j0,k0]
        U[i1*m+j0][col] = 1
        U[i1*m+j1][col] = 1
        V[j0*p+k0][col] = 1
        W[k0*n+i1][col] = 1
        W[k1*n+i1][col] = -1
        col += 1
        # M3 = A[i0,j0] * (B[j0,k1] - B[j1,k1])
        U[i0*m+j0][col] = 1
        V[j0*p+k1][col] = 1
        V[j1*p+k1][col] = -1
        W[k1*n+i0][col] = 1
        W[k1*n+i1][col] = 1
        col += 1
        # M4 = A[i1,j1] * (B[j1,k0] - B[j0,k0])
        U[i1*m+j1][col] = 1
        V[j1*p+k0][col] = 1
        V[j0*p+k0][col] = -1
        W[k0*n+i0][col] = 1
        W[k0*n+i1][col] = 1
        col += 1
        # M5 = (A[i0,j0] + A[i0,j1]) * B[j1,k1]
        U[i0*m+j0][col] = 1
        U[i0*m+j1][col] = 1
        V[j1*p+k1][col] = 1
        W[k0*n+i0][col] = -1
        W[k1*n+i0][col] = 1
        col += 1
        # M6 = (A[i1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k1])
        U[i1*m+j0][col] = 1
        U[i0*m+j0][col] = -1
        V[j0*p+k0][col] = 1
        V[j0*p+k1][col] = 1
        W[k1*n+i1][col] = 1
        col += 1
        # M7 = (A[i0,j1] - A[i1,j1]) * (B[j1,k0] + B[j1,k1])
        U[i0*m+j1][col] = 1
        U[i1*m+j1][col] = -1
        V[j1*p+k0][col] = 1
        V[j1*p+k1][col] = 1
        W[k0*n+i0][col] = 1
        col += 1
        return col
    
    def add_schoolbook(col, i, j, k):
        U[i*m+j][col] = 1
        V[j*p+k][col] = 1
        W[k*n+i][col] = 1
        return col + 1
    
    col = 0
    # Strassen blocks for rows {0,1}, inner {0,1} and {2,3}, outputs {0,1}, {2,3}, {4,5}
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            col = add_strassen_block(col, 0, 1, j0, j0+1, k0, k0+1)
    
    # Schoolbook for row {2}
    for j in range(m):
        for k in range(p):
            col = add_schoolbook(col, 2, j, k)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
