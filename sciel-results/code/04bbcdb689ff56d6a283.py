import numpy as np
import time

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    U_total = np.zeros((4, 11))
    V_total = np.zeros((6, 11))
    W_total = np.zeros((6, 11))
    
    # Strassen for C_left (cols 0,1 of B)
    # A = [[a00, a01], [a10, a11]]
    # B_left = [[b00, b01], [b10, b11]]
    # C_left = [[C00, C01], [C10, C11]]
    
    # M1 = (a00+a11)(b00+b11)
    U_total[0, 0] = 1
    U_total[3, 0] = 1
    V_total[0, 0] = 1
    V_total[4, 0] = 1
    W_total[0, 0] = 1
    W_total[3, 0] = 1
    
    # M2 = (a10+a11)b00
    U_total[2, 1] = 1
    U_total[3, 1] = 1
    V_total[0, 1] = 1
    W_total[1, 1] = 1
    W_total[3, 1] = -1
    
    # M3 = a00(b01-b11)
    U_total[0, 2] = 1
    V_total[1, 2] = 1
    V_total[4, 2] = -1
    W_total[2, 2] = 1
    W_total[3, 2] = 1
    
    # M4 = a11(b10-b00)
    U_total[3, 3] = 1
    V_total[3, 3] = 1
    V_total[0, 3] = -1
    W_total[0, 3] = -1
    W_total[1, 3] = 1
    
    # M5 = (a00+a01)b11
    U_total[0, 4] = 1
    U_total[1, 4] = 1
    V_total[4, 4] = 1
    W_total[0, 4] = 1
    
    # M6 = (a10-a00)(b00+b01)
    U_total[2, 5] = 1
    U_total[0, 5] = -1
    V_total[0, 5] = 1
    V_total[1, 5] = 1
    W_total[3, 5] = 1
    
    # M7 = (a01-a11)(b10+b11)
    U_total[1, 6] = 1
    U_total[3, 6] = -1
    V_total[3, 6] = 1
    V_total[4, 6] = 1
    W_total[0, 6] = -1
    W_total[2, 6] = 1
    
    # Schoolbook for C_right (col 2 of B)
    # C02 = a00*b02 + a01*b12
    # C12 = a10*b02 + a11*b12
    
    # Col 7: a00*b02
    U_total[0, 7] = 1
    V_total[2, 7] = 1
    W_total[4, 7] = 1
    
    # Col 8: a01*b12
    U_total[1, 8] = 1
    V_total[5, 8] = 1
    W_total[4, 8] = 1
    
    # Col 9: a10*b02
    U_total[2, 9] = 1
    V_total[2, 9] = 1
    W_total[5, 9] = 1
    
    # Col 10: a11*b12
    U_total[3, 10] = 1
    V_total[5, 10] = 1
    W_total[5, 10] = 1

    # Verify
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    T_calc = np.einsum('ir,jr,kr->ijk', U_total, V_total, W_total)
    
    if not np.allclose(T_calc, T_target, atol=1e-9):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_sb = np.zeros((n * m, rank_school))
        V_sb = np.zeros((m * p, rank_school))
        W_sb = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
