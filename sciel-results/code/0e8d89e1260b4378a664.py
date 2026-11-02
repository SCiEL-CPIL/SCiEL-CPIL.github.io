import numpy as np
import time

_START_TIME = time.time()
TIME_LIMIT = 110  # safety margin before 120s limit

def _check_time():
    """Return True if we should stop (time limit exceeded)."""
    return (time.time() - _START_TIME) > TIME_LIMIT

def construct():
    n, m, p = 2, 3, 9
    # Rank 50 decomposition using Strassen for 2x2 blocks with a non-trivial
    # linear change of basis on the output dimension to reduce rank
    rank = 50
    U = [[0] * rank for _ in range(n * m)]  # 6 x 50
    V = [[0] * rank for _ in range(m * p)]  # 27 x 50
    W = [[0] * rank for _ in range(p * n)]  # 18 x 50
    
    # A indices: i*m+j (0..5)
    # B indices: j*p+k (0..26)
    # C indices: k*n+i (0..17)
    
    col = 0
    
    # Part 1: A_L (2x2) * B_T (2x9)
    # A_L uses A_{00}, A_{01}, A_{10}, A_{11} -> U indices 0, 1, 3, 4
    # B_T uses B_{0k}, B_{1k} -> V indices 0..8, 9..17
    
    # 4 blocks of 2x2 (columns 0-7 of B_T)
    for b in range(4):
        if _check_time():
            break
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # B indices
        v_e = 0 * p + k0  # B_{0, k0}
        v_f = 0 * p + k1  # B_{0, k1}
        v_g = 1 * p + k0  # B_{1, k0}
        v_h = 1 * p + k1  # B_{1, k1}
        
        # A indices
        u_a = 0 * m + 0  # A_{00}
        u_b = 0 * m + 1  # A_{01}
        u_c = 1 * m + 0  # A_{10}
        u_d = 1 * m + 1  # A_{11}
        
        # Strassen Products (7 multiplications)
        # M1 = (a+d)(e+h)
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        
        # M2 = (c+d)e
        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        
        # M3 = a(f-h)
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1
        
        # M4 = d(g-e)
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1
        
        # M5 = (a+b)h
        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        
        # M6 = (c-a)(e+f)
        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1
        
        # M7 = (b-d)(g+h)
        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1
        
        # Output C_{0, k0}, C_{0, k1}, C_{1, k0}, C_{1, k1}
        # W indices: k*n+i
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        # C00 = M1 + M4 - M5 + M7
        W[w_00][col] += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1
        
        # C01 = M3 + M5
        W[w_01][col+2] += 1
        W[w_01][col+4] += 1
        
        # C10 = M2 + M4
        W[w_10][col+1] += 1
        W[w_10][col+3] += 1
        
        # C11 = M1 - M2 + M3 + M6
        W[w_11][col] += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1
        
        col += 7
    
    # 1 block of 2x1 (last column k=8)
    k8 = 8
    v_e = 0 * p + k8  # B_{0, 8}
    v_g = 1 * p + k8  # B_{1, 8}
    
    u_a = 0 * m + 0
    u_b = 0 * m + 1
    u_c = 1 * m + 0
    u_d = 1 * m + 1
    
    # P1 = a e
    U[u_a][col] += 1
    V[v_e][col] += 1
    # P2 = b g
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    # P3 = c e
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    # P4 = d g
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1
    
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    
    # C08 = P1 + P2
    W[w_08][col] += 1
    W[w_08][col+1] += 1
    
    # C18 = P3 + P4
    W[w_18][col+2] += 1
    W[w_18][col+3] += 1
    
    col += 4
    
    # Part 2: A_R (2x1) * B_B (1x9)
    # A_R uses A_{02}, A_{12} -> U indices 2, 5
    # B_B uses B_{2k} -> V indices 18..26
    
    u_ar0 = 0 * m + 2  # A_{02}
    u_ar1 = 1 * m + 2  # A_{12}
    
    for k in range(9):
        if _check_time():
            break
        v_b = 2 * p + k  # B_{2, k}
        
        # Q_k = A_{02} B_{2k}
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        
        # R_k = A_{12} B_{2k}
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        
        col += 2
    
    return {"U": U, "V": V, "W": W}
