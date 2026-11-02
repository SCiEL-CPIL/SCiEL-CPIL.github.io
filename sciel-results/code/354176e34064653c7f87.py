import numpy as np
import time

DIMENSIONS = [2, 3, 9]
_START_TIME = time.time()
TIME_LIMIT = 110

def _check_time():
    return (time.time() - _START_TIME) > TIME_LIMIT

def construct():
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 4 blocks of 2x2 with rank-6 decomposition
    for b in range(4):
        if _check_time():
            break
        k0 = 2 * b
        k1 = 2 * b + 1
        
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        
        u_a = 0
        u_b = 1
        u_c = 3
        u_d = 4
        
        # 6 terms for 2x2 block
        # M1: (a+d)(e+h)
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        
        # M2: (c+d)(e)
        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        
        # M3: (a)(f-h)
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1
        
        # M4: (d)(g-e)
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1
        
        # M5: (a+b)(h)
        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        
        # M6: (c-a)(e+f)
        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        # w_00: M1 + M4 - M5
        W[w_00][col] += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        
        # w_01: M3 + M5
        W[w_01][col+2] += 1
        W[w_01][col+4] += 1
        
        # w_10: M2 + M4
        W[w_10][col+1] += 1
        W[w_10][col+3] += 1
        
        # w_11: M1 - M2 + M3 - M6
        W[w_11][col] += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] -= 1
        
        col += 6
    
    # 1 block of 2x1
    k8 = 8
    v_e = 0 * p + k8
    v_g = 1 * p + k8
    
    u_a = 0
    u_b = 1
    u_c = 3
    u_d = 4
    
    U[u_a][col] += 1
    V[v_e][col] += 1
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1
    
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    
    W[w_08][col] += 1
    W[w_08][col+1] += 1
    W[w_18][col+2] += 1
    W[w_18][col+3] += 1
    
    col += 4
    
    # Part 2: row 2, all 9 columns
    u_ar0 = 2
    u_ar1 = 5
    
    for k in range(9):
        if _check_time():
            break
        v_b = 2 * p + k
        
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        
        col += 2
    
    return {"U": U, "V": V, "W": W}
