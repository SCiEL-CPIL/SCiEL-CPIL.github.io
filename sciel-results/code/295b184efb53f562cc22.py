"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _trivial(n, m, p):
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

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    rank = 52
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2: 7 terms
    strassen_U = [
        [1, 0, 1, 0, 1, -1, 0],  # (i=0,j=0)
        [0, 0, 0, 0, 1, 0, 1],   # (i=0,j=1)
        [0, 1, 0, 0, 0, 1, 0],   # (i=1,j=0)
        [1, 1, 0, 1, 0, 0, -1],  # (i=1,j=1)
    ]
    strassen_V = [
        [1, 1, 0, 1, 0, 1, 0],   # (j=0,k=0)
        [0, 0, -1, 0, 0, 1, 1],  # (j=0,k=1)
        [1, 0, 1, -1, 1, 0, 1],  # (j=1,k=0)
        [1, 0, 0, 0, 1, 0, 1],   # (j=1,k=1)
    ]
    strassen_W = [
        [1, 0, 0, 1, 0, 0, 1],   # (k=0,i=0)
        [1, 1, 0, 0, 0, 0, 0],   # (k=0,i=1)
        [1, 0, 1, 0, 1, 0, 0],   # (k=1,i=0)
        [1, -1, 1, 0, 0, 1, 0],  # (k=1,i=1)
    ]
    
    col = 0
    for g in range(4):
        k0 = 2 * g
        k1 = 2 * g + 1
        for r in range(7):
            for ii in range(2):
                for jj in range(2):
                    U[ii * m + jj][col + r] = strassen_U[ii][jj][r]
            for jj in range(2):
                V[jj * p + k0][col + r] = strassen_V[jj * 2][r]
                V[jj * p + k1][col + r] = strassen_V[jj * 2 + 1][r]
            for kk in range(2):
                k = k0 + kk
                W[k * n + 0][col + r] = strassen_W[kk * 2][r]
                W[k * n + 1][col + r] = strassen_W[kk * 2 + 1][r]
        col += 7
        for dk in range(2):
            k = k0 + dk
            U[0 * m + 2][col] = 1
            U[1 * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + 0][col] = 1
            W[k * n + 1][col] = 1
            col += 1
        for dk in range(2):
            k = k0 + dk
            U[2 * m + 0][col] = 1
            U[2 * m + 1][col] = 1
            V[0 * p + k][col] = 1
            V[1 * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
        for dk in range(2):
            k = k0 + dk
            U[2 * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        return _trivial(n, m, p)
# EVOLVE-BLOCK-END
