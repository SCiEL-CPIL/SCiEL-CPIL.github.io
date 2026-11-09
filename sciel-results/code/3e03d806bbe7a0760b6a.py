"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 5, 5
    rank = 42
    U = [[0] * rank for _ in range(n * m)]   # 10 x 42
    V = [[0] * rank for _ in range(m * p)]   # 25 x 42
    W = [[0] * rank for _ in range(p * n)]   # 10 x 42
    col = 0

    def add_strassen(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        nonlocal col
        a, b = A_rows
        c, d = A_cols
        e, f = B_rows
        g, h = B_cols
        i, j = C_rows
        k, l = C_cols
        # M1 = (A[a,c]+A[b,d])(B[e,g]+B[f,h]) -> +1 C[i,k], +1 C[j,l]
        U[a*m+c][col] = 1; U[b*m+d][col] = 1
        V[e*p+g][col] = 1; V[f*p+h][col] = 1
        W[k*n+i][col] = 1; W[l*n+j][col] = 1
        col += 1
        # M2 = (A[b,c]+A[b,d])B[e,g] -> +1 C[j,k], -1 C[j,l]
        U[b*m+c][col] = 1; U[b*m+d][col] = 1
        V[e*p+g][col] = 1
        W[k*n+j][col] = 1; W[l*n+j][col] = -1
        col += 1
        # M3 = A[a,c](B[e,h]-B[f,h]) -> +1 C[i,l], +1 C[j,l]
        U[a*m+c][col] = 1
        V[e*p+h][col] = 1; V[f*p+h][col] = -1
        W[l*n+i][col] = 1; W[l*n+j][col] = 1
        col += 1
        # M4 = A[b,d](B[f,g]-B[e,g]) -> +1 C[i,k], +1 C[j,k]
        U[b*m+d][col] = 1
        V[f*p+g][col] = 1; V[e*p+g][col] = -1
        W[k*n+i][col] = 1; W[k*n+j][col] = 1
        col += 1
        # M5 = (A[a,c]+A[a,d])B[f,h] -> -1 C[i,k], +1 C[i,l]
        U[a*m+c][col] = 1; U[a*m+d][col] = 1
        V[f*p+h][col] = 1
        W[k*n+i][col] = -1; W[l*n+i][col] = 1
        col += 1
        # M6 = (A[b,c]-A[a,c])(B[e,g]+B[e,h]) -> +1 C[j,l]
        U[b*m+c][col] = 1; U[a*m+c][col] = -1
        V[e*p+g][col] = 1; V[e*p+h][col] = 1
        W[l*n+j][col] = 1
        col += 1
        # M7 = (A[a,d]-A[b,d])(B[f,g]+B[f,h]) -> +1 C[i,k], +1 C[j,k]
        U[a*m+d][col] = 1; U[b*m+d][col] = -1
        V[f*p+g][col] = 1; V[f*p+h][col] = 1
        W[k*n+i][col] = 1; W[k*n+j][col] = 1
        col += 1

    def add_2x1_1x2(A_row, A_col, B_row, B_col, C_rows, C_cols):
        nonlocal col
        i, j = C_rows
        k, l = C_cols
        U[A_row*m+A_col][col] = 1; V[B_row*p+B_col][col] = 1; W[k*n+i][col] = 1; col += 1
        U[A_row*m+A_col][col] = 1; V[B_row*p+(B_col+1)][col] = 1; W[l*n+i][col] = 1; col += 1
        U[(A_row+1)*m+A_col][col] = 1; V[B_row*p+B_col][col] = 1; W[k*n+j][col] = 1; col += 1
        U[(A_row+1)*m+A_col][col] = 1; V[B_row*p+(B_col+1)][col] = 1; W[l*n+j][col] = 1; col += 1

    def add_2x2_2x1(A_rows, A_cols, B_rows, B_col, C_rows, C_col):
        nonlocal col
        i, j = C_rows
        k = C_col
        U[A_rows[0]*m+A_cols[0]][col] = 1; V[B_rows[0]*p+B_col][col] = 1; W[k*n+i][col] = 1; col += 1
        U[A_rows[0]*m+A_cols[1]][col] = 1; V[B_rows[1]*p+B_col][col] = 1; W[k*n+i][col] = 1; col += 1
        U[A_rows[1]*m+A_cols[0]][col] = 1; V[B_rows[0]*p+B_col][col] = 1; W[k*n+j][col] = 1; col += 1
        U[A_rows[1]*m+A_cols[1]][col] = 1; V[B_rows[1]*p+B_col][col] = 1; W[k*n+j][col] = 1; col += 1

    # C[:,0:2] = A[:,0:2]*B[0:2,0:2] + A[:,2:4]*B[2:4,0:2] + A[:,4]*B[4,0:2]
    add_strassen([0,1],[0,1],[0,1],[0,1],[0,1],[0,1])
    add_strassen([0,1],[2,3],[2,3],[0,1],[0,1],[0,1])
    add_2x1_1x2(0,4,4,0,[0,1],[0,1])

    # C[:,2:4] = A[:,0:2]*B[0:2,2:4] + A[:,2:4]*B[2:4,2:4] + A[:,4]*B[4,2:4]
    add_strassen([0,1],[0,1],[0,1],[2,3],[0,1],[2,3])
    add_strassen([0,1],[2,3],[2,3],[2,3],[0,1],[2,3])
    add_2x1_1x2(0,4,4,2,[0,1],[2,3])

    # C[:,4] = A[:,0:2]*B[0:2,4] + A[:,2:4]*B[2:4,4] + A[:,4]*B[4,4]
    add_2x2_2x1([0,1],[0,1],[0,1],4,[0,1],4)
    add_2x2_2x1([0,1],[2,3],[2,3],4,[0,1],4)
    # A[:,4] * B[4,4] -> C[:,4]: 2x1 * 1x1 -> 2x1, rank 2
    U[0*m+4][col] = 1; V[4*p+4][col] = 1; W[4*n+0][col] = 1; col += 1
    U[1*m+4][col] = 1; V[4*p+4][col] = 1; W[4*n+1][col] = 1; col += 1

    assert col == rank, f"col={col}, rank={rank}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
