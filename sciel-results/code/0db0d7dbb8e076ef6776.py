import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 120.0
_start_time = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _start_time)

def construct():
    n, m, p = 2, 3, 9
    rank = 50
    
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    col = 0
    
    # Part 1: Strassen on 4 blocks of 2x2 (cols 0-7 of B, rows 0-1)
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        # M1 = (a+d)(e+h)
        U[0, col] += 1; U[4, col] += 1
        V[v_e, col] += 1; V[v_h, col] += 1
        # M2 = (c+d)e
        U[3, col+1] += 1; U[4, col+1] += 1
        V[v_e, col+1] += 1
        # M3 = a(f-h)
        U[0, col+2] += 1
        V[v_f, col+2] += 1; V[v_h, col+2] -= 1
        # M4 = d(g-e)
        U[4, col+3] += 1
        V[v_g, col+3] += 1; V[v_e, col+3] -= 1
        # M5 = (a+b)h
        U[0, col+4] += 1; U[1, col+4] += 1
        V[v_h, col+4] += 1
        # M6 = (c-a)(e+f)
        U[3, col+5] += 1; U[0, col+5] -= 1
        V[v_e, col+5] += 1; V[v_f, col+5] += 1
        # M7 = (b-d)(g+h)
        U[1, col+6] += 1; U[4, col+6] -= 1
        V[v_g, col+6] += 1; V[v_h, col+6] += 1
        
        # Reconstruct
        W[w_00, col] += 1
        W[w_00, col+3] += 1
        W[w_00, col+4] -= 1
        W[w_00, col+6] += 1
        W[w_01, col+2] += 1
        W[w_01, col+4] += 1
        W[w_10, col+1] += 1
        W[w_10, col+3] += 1
        W[w_11, col] += 1
        W[w_11, col+1] -= 1
        W[w_11, col+2] += 1
        W[w_11, col+5] += 1
        
        col += 7
    
    # Part 2: Naive for B col 8, rows 0-1
    k8 = 8
    v_e8 = 0 * p + k8
    v_g8 = 1 * p + k8
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    
    U[0, col] += 1; V[v_e8, col] += 1; W[w_08, col] += 1
    U[1, col+1] += 1; V[v_g8, col+1] += 1; W[w_08, col+1] += 1
    U[3, col+2] += 1; V[v_e8, col+2] += 1; W[w_18, col+2] += 1
    U[4, col+3] += 1; V[v_g8, col+3] += 1; W[w_18, col+3] += 1
    
    col += 4
    
    # Part 3: A_right (col 2) x B_bottom (row 2)
    u_ar0 = 0 * m + 2
    u_ar1 = 1 * m + 2
    
    for k in range(p):
        v_bk = 2 * p + k
        w_0k = k * n + 0
        w_1k = k * n + 1
        
        U[u_ar0, col] += 1
        V[v_bk, col] += 1
        W[w_0k, col] += 1
        
        U[u_ar1, col+1] += 1
        V[v_bk, col+1] += 1
        W[w_1k, col+1] += 1
        
        col += 2
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
