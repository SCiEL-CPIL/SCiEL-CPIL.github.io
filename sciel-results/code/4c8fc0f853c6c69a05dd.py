"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 10
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Term 1: A_col0 * B_row0 (2x1x5) -> Rank 1
    # U: rows i*3+0
    # V: rows 0*5+k
    # W: rows k*2+i
    col_idx = 0
    for i in range(n):
        U[i * m + 0][col_idx] = 1
    for k in range(p):
        V[0 * p + k][col_idx] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][col_idx] = 1
    col_idx += 1

    # Term 2: A_cols12 * B_rows12_cols01 (2x2x2) -> Rank 7 (Strassen)
    # Local A: A[i, j+1]
    # Local B: B[j+1, k]
    # Local C: C[i, k]
    # Strassen factors:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a-b)(g+h)
    # M4 = (d)(f-h)
    # M5 = (a+c)(h)
    # M6 = (b-c)(e+g)
    # M7 = (b+d)(f)
    
    # a=A01, b=A02, c=A11, d=A12
    # e=B10, f=B11, g=B20, h=B21
    
    # M1
    U[0 * m + 1][col_idx] = 1
    U[1 * m + 2][col_idx] = 1
    V[1 * p + 0][col_idx] = 1
    V[2 * p + 1][col_idx] = 1
    # C00 += 1, C01 += 0, C10 += 0, C11 += 1
    W[0 * n + 0][col_idx] = 1
    W[1 * n + 1][col_idx] = 1
    col_idx += 1
    
    # M2
    U[1 * m + 1][col_idx] = 1
    U[1 * m + 2][col_idx] = 1
    V[1 * p + 0][col_idx] = 1
    # C10 += 1
    W[0 * n + 1][col_idx] = 1
    col_idx += 1
    
    # M3
    U[0 * m + 1][col_idx] = 1
    U[0 * m + 2][col_idx] = -1
    V[2 * p + 0][col_idx] = 1
    V[2 * p + 1][col_idx] = 1
    # C01 += 1, C11 += 1
    W[1 * n + 0][col_idx] = 1
    W[1 * n + 1][col_idx] = 1
    col_idx += 1
    
    # M4
    U[1 * m + 2][col_idx] = 1
    V[1 * p + 1][col_idx] = 1
    V[2 * p + 1][col_idx] = -1
    # C00 += 1, C10 += 1
    W[0 * n + 0][col_idx] = 1
    W[0 * n + 1][col_idx] = 1
    col_idx += 1
    
    # M5
    U[0 * m + 1][col_idx] = 1
    U[1 * m + 1][col_idx] = 1
    V[2 * p + 1][col_idx] = 1
    # C00 -= 1, C01 += 1
    W[0 * n + 0][col_idx] = -1
    W[1 * n + 0][col_idx] = 1
    col_idx += 1
    
    # M6
    U[0 * m + 2][col_idx] = 1
    U[1 * m + 1][col_idx] = -1
    V[1 * p + 0][col_idx] = 1
    V[2 * p + 0][col_idx] = 1
    # C11 += 1
    W[1 * n + 1][col_idx] = 1
    col_idx += 1
    
    # M7
    U[0 * m + 2][col_idx] = 1
    U[1 * m + 2][col_idx] = 1
    V[1 * p + 1][col_idx] = 1
    # C00 += 1, C01 += 0? No C01?
    # C00 += 1
    W[0 * n + 0][col_idx] = 1
    col_idx += 1

    # Term 3: A_col1 * B_row1_cols24 (2x1x3) -> Rank 1
    for i in range(n):
        U[i * m + 1][col_idx] = 1
    for k in range(2, p):
        V[1 * p + k][col_idx] = 1
    for k in range(2, p):
        for i in range(n):
            W[k * n + i][col_idx] = 1
    col_idx += 1

    # Term 4: A_col2 * B_row2_cols24 (2x1x3) -> Rank 1
    for i in range(n):
        U[i * m + 2][col_idx] = 1
    for k in range(2, p):
        V[2 * p + k][col_idx] = 1
    for k in range(2, p):
        for i in range(n):
            W[k * n + i][col_idx] = 1
    col_idx += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
