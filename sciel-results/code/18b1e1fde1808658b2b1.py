"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 26 decomposition
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def idx(i, j):
        return i * m + j

    def widx(i, j):
        return j * n + i

    col = 0
    
    # Strassen 2x2 core (indices 0,1 for i,j)
    # M1 = (a11+a22)(b11+b22) -> C11, C22
    # C11 += M1, C22 += M1
    # M1
    U[idx(0,0)][col] = 1
    U[idx(1,1)][col] = 1
    V[idx(0,0)][col] = 1
    V[idx(1,1)][col] = 1
    W[widx(0,0)][col] = 1
    W[widx(1,1)][col] = 1
    col += 1
    
    # M2 = (a21+a22)b11 -> C21
    # C21 -= M2
    U[idx(1,0)][col] = 1
    U[idx(1,1)][col] = 1
    V[idx(0,0)][col] = 1
    W[widx(0,1)][col] = -1
    col += 1
    
    # M3 = a11(b12-b22) -> C12, C22
    # C12 += M3, C22 -= M3
    U[idx(0,0)][col] = 1
    V[idx(0,1)][col] = 1
    V[idx(1,1)][col] = -1
    W[widx(1,0)][col] = 1
    W[widx(1,1)][col] = -1
    col += 1
    
    # M4 = a22(b21-b11) -> C11, C21, C22
    # C11 += M4, C21 += M4, C22 -= M4
    U[idx(1,1)][col] = 1
    V[idx(1,0)][col] = 1
    V[idx(0,0)][col] = -1
    W[widx(0,0)][col] = 1
    W[widx(0,1)][col] = 1
    W[widx(1,1)][col] = -1
    col += 1
    
    # M5 = (a11+a12)b22 -> C11, C12
    # C11 -= M5, C12 += M5
    U[idx(0,0)][col] = 1
    U[idx(0,1)][col] = 1
    V[idx(1,1)][col] = 1
    W[widx(0,0)][col] = -1
    W[widx(1,0)][col] = 1
    col += 1
    
    # M6 = (a21-a11)(b11+b12) -> C21, C22
    # C21 += M6, C22 += M6
    U[idx(1,0)][col] = 1
    U[idx(0,0)][col] = -1
    V[idx(0,0)][col] = 1
    V[idx(0,1)][col] = 1
    W[widx(0,1)][col] = 1
    W[widx(1,1)][col] = 1
    col += 1
    
    # M7 = (a12-a22)(b21+b22) -> C11, C12, C21
    # C11 += M7, C12 += M7, C21 += M7
    U[idx(0,1)][col] = 1
    U[idx(1,1)][col] = -1
    V[idx(1,0)][col] = 1
    V[idx(1,1)][col] = 1
    W[widx(0,0)][col] = 1
    W[widx(1,0)][col] = 1
    W[widx(0,1)][col] = 1
    col += 1

    # Border terms k=3, i,j in 0,1
    # a13 b31 -> C11
    U[idx(0,2)][col] = 1
    V[idx(2,0)][col] = 1
    W[widx(0,0)][col] = 1
    col += 1
    
    # a13 b32 -> C12
    U[idx(0,2)][col] = 1
    V[idx(2,1)][col] = 1
    W[widx(1,0)][col] = 1
    col += 1
    
    # a23 b31 -> C21
    U[idx(1,2)][col] = 1
    V[idx(2,0)][col] = 1
    W[widx(0,1)][col] = 1
    col += 1
    
    # a23 b32 -> C22
    U[idx(1,2)][col] = 1
    V[idx(2,1)][col] = 1
    W[widx(1,1)][col] = 1
    col += 1

    # i=2 (row 3)
    # a31 b11 -> C31
    U[idx(2,0)][col] = 1
    V[idx(0,0)][col] = 1
    W[widx(0,2)][col] = 1
    col += 1
    
    # a31 b12 -> C32
    U[idx(2,0)][col] = 1
    V[idx(0,1)][col] = 1
    W[widx(1,2)][col] = 1
    col += 1
    
    # a32 b21 -> C31
    U[idx(2,1)][col] = 1
    V[idx(1,0)][col] = 1
    W[widx(0,2)][col] = 1
    col += 1
    
    # a32 b22 -> C32
    U[idx(2,1)][col] = 1
    V[idx(1,1)][col] = 1
    W[widx(1,2)][col] = 1
    col += 1
    
    # a33 b31 -> C31
    U[idx(2,2)][col] = 1
    V[idx(2,0)][col] = 1
    W[widx(0,2)][col] = 1
    col += 1
    
    # a33 b32 -> C32
    U[idx(2,2)][col] = 1
    V[idx(2,1)][col] = 1
    W[widx(1,2)][col] = 1
    col += 1

    # j=2 (col 3)
    # a11 b13 -> C13
    U[idx(0,0)][col] = 1
    V[idx(0,2)][col] = 1
    W[widx(2,0)][col] = 1
    col += 1
    
    # a12 b23 -> C13
    U[idx(0,1)][col] = 1
    V[idx(1,2)][col] = 1
    W[widx(2,0)][col] = 1
    col += 1
    
    # a21 b13 -> C23
    U[idx(1,0)][col] = 1
    V[idx(0,2)][col] = 1
    W[widx(2,1)][col] = 1
    col += 1
    
    # a22 b23 -> C23
    U[idx(1,1)][col] = 1
    V[idx(1,2)][col] = 1
    W[widx(2,1)][col] = 1
    col += 1
    
    # a13 b33 -> C13
    U[idx(0,2)][col] = 1
    V[idx(2,2)][col] = 1
    W[widx(2,0)][col] = 1
    col += 1
    
    # a23 b33 -> C23
    U[idx(1,2)][col] = 1
    V[idx(2,2)][col] = 1
    W[widx(2,1)][col] = 1
    col += 1

    # C33
    # a31 b13 -> C33
    U[idx(2,0)][col] = 1
    V[idx(0,2)][col] = 1
    W[widx(2,2)][col] = 1
    col += 1
    
    # a32 b23 -> C33
    U[idx(2,1)][col] = 1
    V[idx(1,2)][col] = 1
    W[widx(2,2)][col] = 1
    col += 1
    
    # a33 b33 -> C33
    U[idx(2,2)][col] = 1
    V[idx(2,2)][col] = 1
    W[widx(2,2)][col] = 1
    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
