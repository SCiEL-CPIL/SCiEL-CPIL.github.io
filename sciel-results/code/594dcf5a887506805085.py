import numpy as np

def construct():
    n, m, p = 2, 4, 5
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(a_row, a_col, b_row, b_col, c_row, c_col):
        nonlocal col
        idx_p00 = a_row * m + a_col
        idx_p01 = a_row * m + (a_col + 1)
        idx_p10 = (a_row + 1) * m + a_col
        idx_p11 = (a_row + 1) * m + (a_col + 1)
        
        idx_q00 = b_row * p + b_col
        idx_q01 = b_row * p + (b_col + 1)
        idx_q10 = (b_row + 1) * p + b_col
        idx_q11 = (b_row + 1) * p + (b_col + 1)
        
        idx_c00 = c_col * n + c_row
        idx_c01 = (c_col + 1) * n + c_row
        idx_c10 = c_col * n + (c_row + 1)
        idx_c11 = (c_col + 1) * n + (c_row + 1)
        
        # M1: (p00+p11)(q00+q11) -> C00(+1), C11(+1)
        U[idx_p00][col] += 1; U[idx_p11][col] += 1
        V[idx_q00][col] += 1; V[idx_q11][col] += 1
        W[idx_c00][col] += 1; W[idx_c11][col] += 1
        col += 1
        
        # M2: (p10+p11)q00 -> C10(+1), C11(-1)
        U[idx_p10][col] += 1; U[idx_p11][col] += 1
        V[idx_q00][col] += 1
        W[idx_c10][col] += 1; W[idx_c11][col] -= 1
        col += 1
        
        # M3: p00(q01-q11) -> C01(+1), C11(+1)
        U[idx_p00][col] += 1
        V[idx_q01][col] += 1; V[idx_q11][col] -= 1
        W[idx_c01][col] += 1; W[idx_c11][col] += 1
        col += 1
        
        # M4: p11(q10-q00) -> C00(+1), C10(+1)
        U[idx_p11][col] += 1
        V[idx_q10][col] += 1; V[idx_q00][col] -= 1
        W[idx_c00][col] += 1; W[idx_c10][col] += 1
        col += 1
        
        # M5: (p00+p01)q11 -> C00(-1), C01(+1)
        U[idx_p00][col] += 1; U[idx_p01][col] += 1
        V[idx_q11][col] += 1
        W[idx_c00][col] -= 1; W[idx_c01][col] += 1
        col += 1
        
        # M6: (p10-p00)(q00+q01) -> C11(+1)
        U[idx_p10][col] += 1; U[idx_p00][col] -= 1
        V[idx_q00][col] += 1; V[idx_q01][col] += 1
        W[idx_c11][col] += 1
        col += 1
        
        # M7: (p01-p11)(q10+q11) -> C00(+1)
        U[idx_p01][col] += 1; U[idx_p11][col] -= 1
        V[idx_q10][col] += 1; V[idx_q11][col] += 1
        W[idx_c00][col] += 1
        col += 1

    # 4 Strassen blocks for k=0..3
    add_strassen(0, 0, 0, 0, 0, 0)  # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    add_strassen(0, 2, 2, 0, 0, 0)  # A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    add_strassen(0, 0, 0, 2, 0, 2)  # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    add_strassen(0, 2, 2, 2, 0, 2)  # A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    
    # 2×4×1 block for k=4: C[i] += A[i,j]*B[j,4] for j=0..3
    # 4 terms, each A[i,j]*B[j,4] -> C[i,4]
    for j in range(4):
        for i in range(2):
            U[i * m + j][col] += 1
            V[j * p + 4][col] += 1
            W[4 * n + i][col] += 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
