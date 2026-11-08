import time

DIMENSIONS = [4, 4, 6]

def _strassen_2x2x2():
    """Return U, V, W (each 4x7) for rank-7 Strassen decomposition of 2x2x2 matmul."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1; V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1; V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00(B10-B11)
    U[0][2] = 1; V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B01-B10)
    U[3][3] = 1; V[1][3] = 1; V[2][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1; V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B10)
    U[2][5] = 1; U[0][5] = -1; V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B01+B11)
    U[1][6] = 1; U[3][6] = -1; V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    rank_444 = 49
    rank_442 = 32
    total_rank = rank_444 + rank_442

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    U_s, V_s, W_s = _strassen_2x2x2()

    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for a in range(2):
                for b in range(2):
                    for c in range(2):
                        for d in range(2):
                            u_val = U_s[a * 2 + c][r1] * U_s[b * 2 + d][r2]
                            if u_val != 0:
                                row_u = (2 * a + b) * 4 + (2 * c + d)
                                U[row_u][r] = u_val
                        for e in range(2):
                            for f in range(2):
                                v_val = V_s[c * 2 + e][r1] * V_s[d * 2 + f][r2]
                                if v_val != 0:
                                    row_v = (2 * c + d) * 6 + (2 * e + f)
                                    V[row_v][r] = v_val
                                w_val = W_s[e * 2 + a][r1] * W_s[f * 2 + b][r2]
                                if w_val != 0:
                                    row_w = (2 * e + f) * 4 + (2 * a + b)
                                    W[row_w][r] = w_val

    col_offset = rank_444
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U[i * m + j][col_offset + col] = 1
                V[j * p + k][col_offset + col] = 1
                W[k * n + i][col_offset + col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
