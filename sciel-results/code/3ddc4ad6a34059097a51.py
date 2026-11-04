import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # 120s limit, 20s safety margin

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
        
        # M1
        U[idx_p00][col] += 1; U[idx_p11][col] += 1
        V[idx_q00][col] += 1; V[idx_q11][col] += 1
        W[idx_c00][col] += 1; W[idx_c11][col] += 1
        col += 1
        # M2
        U[idx_p10][col] += 1; U[idx_p11][col] += 1
        V[idx_q00][col] += 1
        W[idx_c10][col] += 1; W[idx_c11][col] -= 1
        col += 1
        # M3
        U[idx_p00][col] += 1
        V[idx_q01][col] += 1; V[idx_q11][col] -= 1
        W[idx_c01][col] += 1; W[idx_c11][col] += 1
        col += 1
        # M4
        U[idx_p11][col] += 1
        V[idx_q10][col] += 1; V[idx_q00][col] -= 1
        W[idx_c00][col] += 1; W[idx_c10][col] += 1
        col += 1
        # M5
        U[idx_p00][col] += 1; U[idx_p01][col] += 1
        V[idx_q11][col] += 1
        W[idx_c00][col] -= 1; W[idx_c01][col] += 1
        col += 1
        # M6
        U[idx_p10][col] += 1; U[idx_p00][col] -= 1
        V[idx_q00][col] += 1; V[idx_q01][col] += 1
        W[idx_c11][col] += 1
        col += 1
        # M7
        U[idx_p01][col] += 1; U[idx_p11][col] -= 1
        V[idx_q10][col] += 1; V[idx_q11][col] += 1
        W[idx_c00][col] += 1
        col += 1

    def add_schoolbook_221(a_row, a_col, b_row, b_col, c_row, c_col):
        nonlocal col
        idx_p00 = a_row * m + a_col
        idx_p01 = a_row * m + (a_col + 1)
        idx_p10 = (a_row + 1) * m + a_col
        idx_p11 = (a_row + 1) * m + (a_col + 1)
        
        idx_q0 = b_row * p + b_col
        idx_q1 = (b_row + 1) * p + b_col
        
        idx_c0 = c_col * n + c_row
        idx_c1 = c_col * n + (c_row + 1)
        
        U[idx_p00][col] += 1; V[idx_q0][col] += 1; W[idx_c0][col] += 1
        col += 1
        U[idx_p01][col] += 1; V[idx_q1][col] += 1; W[idx_c0][col] += 1
        col += 1
        U[idx_p10][col] += 1; V[idx_q0][col] += 1; W[idx_c1][col] += 1
        col += 1
        U[idx_p11][col] += 1; V[idx_q1][col] += 1; W[idx_c1][col] += 1
        col += 1

    # 4 Strassen blocks
    add_strassen(0, 0, 0, 0, 0, 0)
    add_strassen(0, 2, 2, 0, 0, 0)
    add_strassen(0, 0, 0, 2, 0, 2)
    add_strassen(0, 2, 2, 2, 0, 2)
    
    # 2 Schoolbook blocks
    add_schoolbook_221(0, 0, 0, 4, 0, 4)
    add_schoolbook_221(0, 2, 2, 4, 0, 4)
    
    return {"U": U, "V": V, "W": W}
