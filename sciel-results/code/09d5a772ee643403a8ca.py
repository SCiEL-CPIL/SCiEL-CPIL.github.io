"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    rank = 19
    U = [[0] * rank for _ in range(n * m)]  # (6, 19)
    V = [[0] * rank for _ in range(m * p)]  # (15, 19)
    W = [[0] * rank for _ in range(p * n)]  # (10, 19)
    
    # Term 0: k=0 part
    for i in range(n):
        U[i * m + 0][0] = 1
    for k in range(p):
        V[0 * p + k][0] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][0] = 1
    
    def add_strassen_2x2_2x2(col_start, k_cols):
        k0, k1 = k_cols
        c = col_start
        # m1 = (p00+p11)*(q00+q11)
        U[0*m+1][c] = 1; U[1*m+2][c] = 1
        V[1*p+k0][c] = 1; V[2*p+k1][c] = 1
        W[k0*n+0][c] = 1; W[k1*n+0][c] = 1; W[k0*n+1][c] = 1; W[k1*n+1][c] = 1
        c += 1
        # m2 = (p10+p11)*q00
        U[1*m+1][c] = 1; U[1*m+2][c] = 1
        V[1*p+k0][c] = 1
        W[k0*n+1][c] = 1
        c += 1
        # m3 = p00*(q01-q11)
        U[0*m+1][c] = 1
        V[1*p+k1][c] = 1; V[2*p+k1][c] = -1
        W[k1*n+0][c] = 1
        c += 1
        # m4 = p11*(q10-q00)
        U[1*m+2][c] = 1
        V[2*p+k0][c] = 1; V[1*p+k0][c] = -1
        W[k0*n+1][c] = 1; W[k1*n+1][c] = 1
        c += 1
        # m5 = (p00+p01)*q11
        U[0*m+1][c] = 1; U[0*m+2][c] = 1
        V[2*p+k1][c] = 1
        W[k1*n+0][c] = 1; W[k1*n+1][c] = -1
        c += 1
        # m6 = (p10-p00)*(q00+q01)
        U[1*m+1][c] = 1; U[0*m+1][c] = -1
        V[1*p+k0][c] = 1; V[1*p+k1][c] = 1
        W[k0*n+1][c] = 1
        c += 1
        # m7 = (p01-p11)*(q10+q11)
        U[0*m+2][c] = 1; U[1*m+2][c] = -1
        V[2*p+k0][c] = 1; V[2*p+k1][c] = 1
        W[k1*n+0][c] = 1
        c += 1
        return c
    
    col = add_strassen_2x2_2x2(1, [0, 1])
    col = add_strassen_2x2_2x2(col, [2, 3])
    
    # 2x2 x 2x1 for column 4: 4 terms
    U[0*m+1][col] = 1; V[1*p+4][col] = 1; W[4*n+0][col] = 1; col += 1
    U[1*m+1][col] = 1; V[1*p+4][col] = 1; W[4*n+1][col] = 1; col += 1
    U[0*m+2][col] = 1; V[2*p+4][col] = 1; W[4*n+0][col] = 1; col += 1
    U[1*m+2][col] = 1; V[2*p+4][col] = 1; W[4*n+1][col] = 1; col += 1
    
    assert col == rank
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
