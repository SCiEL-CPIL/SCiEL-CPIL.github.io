import numpy as np
import time

_START_TIME = time.time()
TIME_LIMIT = 110

def _check_time():
    return (time.time() - _START_TIME) > TIME_LIMIT

def construct():
    n, m, p = 2, 3, 9
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: First 3 columns of B (indices 0, 1, 2)
    # Uses Rank-17 decomposition for 2x3x3 multiplication
    # A_L (2x2) * B_T (2x3) + A_R (2x1) * B_B (1x3)
    
    # A indices: 0,1,2,3,4,5
    # B indices: 0..26
    # W indices: 0..17
    
    # A_L: A[0,0]=U0, A[0,1]=U1, A[1,0]=U3, A[1,1]=U4
    # A_R: A[0,2]=U2, A[1,2]=U5
    
    # B_T for cols 0,1,2:
    # B[0,0]=V0, B[0,1]=V1, B[0,2]=V2
    # B[1,0]=V9, B[1,1]=V10, B[1,2]=V11
    # B_B for cols 0,1,2:
    # B[2,0]=V18, B[2,1]=V19, B[2,2]=V20
    
    # --- A_L * B_T (2x2x3) ---
    # Split B_T into B_T1 (cols 0,1) and B_T2 (col 2)
    
    # B_T1: V0, V1, V9, V10
    # Strassen for A_L * B_T1 (Rank 7)
    u_a, u_b, u_c, u_d = 0, 1, 3, 4
    v_e, v_f, v_g, v_h = 0, 1, 9, 10
    
    # M1 = (a+d)(e+h)
    U[u_a][col] += 1; U[u_d][col] += 1
    V[v_e][col] += 1; V[v_h][col] += 1
    # M2 = (c+d)e
    U[u_c][col+1] += 1; U[u_d][col+1] += 1
    V[v_e][col+1] += 1
    # M3 = a(f-h)
    U[u_a][col+2] += 1
    V[v_f][col+2] += 1; V[v_h][col+2] -= 1
    # M4 = d(g-e)
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1; V[v_e][col+3] -= 1
    # M5 = (a+b)h
    U[u_a][col+4] += 1; U[u_b][col+4] += 1
    V[v_h][col+4] += 1
    # M6 = (c-a)(e+f)
    U[u_c][col+5] += 1; U[u_a][col+5] -= 1
    V[v_e][col+5] += 1; V[v_f][col+5] += 1
    # M7 = (b-d)(g+h)
    U[u_b][col+6] += 1; U[u_d][col+6] -= 1
    V[v_g][col+6] += 1; V[v_h][col+6] += 1
    
    # Output for B_T1 (cols 0, 1)
    w_00, w_01, w_10, w_11 = 0, 1, 2, 3 # k*n+i: 0*2+0, 1*2+0, 0*2+1, 1*2+1
    
    # C00 (k=0, i=0) = M1 + M4 - M5 + M7
    W[w_00][col] += 1; W[w_00][col+3] += 1
    W[w_00][col+4] -= 1; W[w_00][col+6] += 1
    # C01 (k=1, i=0) = M3 + M5
    W[w_01][col+2] += 1; W[w_01][col+4] += 1
    # C10 (k=0, i=1) = M2 + M4
    W[w_10][col+1] += 1; W[w_10][col+3] += 1
    # C11 (k=1, i=1) = M1 - M2 + M3 + M6
    W[w_11][col] += 1; W[w_11][col+1] -= 1
    W[w_11][col+2] += 1; W[w_11][col+5] += 1
    
    col += 7
    
    # B_T2: V2 (B[0,2]), V11 (B[1,2])
    # Schoolbook for A_L * B_T2 (Rank 4)
    v_k, v_l = 2, 11
    
    # P1 = a * v_k
    U[u_a][col] += 1; V[v_k][col] += 1
    # P2 = b * v_l
    U[u_b][col+1] += 1; V[v_l][col+1] += 1
    # P3 = c * v_k
    U[u_c][col+2] += 1; V[v_k][col+2] += 1
    # P4 = d * v_l
    U[u_d][col+3] += 1; V[v_l][col+3] += 1
    
    # Output for B_T2 (col 2)
    w_02 = 2 * n + 0 # k=2, i=0
    w_12 = 2 * n + 1 # k=2, i=1
    
    # C02 = P1 + P2
    W[w_02][col] += 1; W[w_02][col+1] += 1
    # C12 = P3 + P4
    W[w_12][col+2] += 1; W[w_12][col+3] += 1
    
    col += 4
    
    # --- A_R * B_B (2x1x3) ---
    # Schoolbook (Rank 6)
    u_ar0, u_ar1 = 2, 5
    v_b0, v_b1, v_b2 = 18, 19, 20
    
    # Q0 = A[0,2] * B[2,0]
    U[u_ar0][col] += 1; V[v_b0][col] += 1
    W[0][col] += 1 # k=0, i=0
    # R0 = A[1,2] * B[2,0]
    U[u_ar1][col+1] += 1; V[v_b0][col+1] += 1
    W[1][col+1] += 1 # k=0, i=1
    
    # Q1 = A[0,2] * B[2,1]
    U[u_ar0][col+2] += 1; V[v_b1][col+2] += 1
    W[2][col+2] += 1 # k=1, i=0
    # R1 = A[1,2] * B[2,1]
    U[u_ar1][col+3] += 1; V[v_b1][col+3] += 1
    W[3][col+3] += 1 # k=1, i=1
    
    # Q2 = A[0,2] * B[2,2]
    U[u_ar0][col+4] += 1; V[v_b2][col+4] += 1
    W[4][col+4] += 1 # k=2, i=0
    # R2 = A[1,2] * B[2,2]
    U[u_ar1][col+5] += 1; V[v_b2][col+5] += 1
    W[5][col+5] += 1 # k=2, i=1
    
    col += 6
    
    # Part 2: Remaining 6 columns of B (indices 3..8)
    # A_L (2x2) * B_T_rem (2x6) + A_R (2x1) * B_B_rem (1x6)
    
    # A_L * B_T_rem: 3 blocks of 2 columns
    for b in range(3):
        if _check_time():
            break
        k0 = 3 + 2 * b
        k1 = 4 + 2 * b
        
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        
        u_a, u_b, u_c, u_d = 0, 1, 3, 4
        
        # Strassen
        U[u_a][col] += 1; U[u_d][col] += 1
        V[v_e][col] += 1; V[v_h][col] += 1
        U[u_c][col+1] += 1; U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1; V[v_h][col+2] -= 1
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1; V[v_e][col+3] -= 1
        U[u_a][col+4] += 1; U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        U[u_c][col+5] += 1; U[u_a][col+5] -= 1
        V[v_e][col+5] += 1; V[v_f][col+5] += 1
        U[u_b][col+6] += 1; U[u_d][col+6] -= 1
        V[v_g][col+6] += 1; V[v_h][col+6] += 1
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        W[w_00][col] += 1; W[w_00][col+3] += 1
        W[w_00][col+4] -= 1; W[w_00][col+6] += 1
        W[w_01][col+2] += 1; W[w_01][col+4] += 1
        W[w_10][col+1] += 1; W[w_10][col+3] += 1
        W[w_11][col] += 1; W[w_11][col+1] -= 1
        W[w_11][col+2] += 1; W[w_11][col+5] += 1
        
        col += 7
        
    # A_R * B_B_rem: 6 columns
    u_ar0, u_ar1 = 2, 5
    for k in range(3, 9):
        if _check_time():
            break
        v_b = 2 * p + k
        
        U[u_ar0][col] += 1; V[v_b][col] += 1
        W[k * n + 0][col] += 1
        
        U[u_ar1][col+1] += 1; V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        
        col += 2

    return {"U": U, "V": V, "W": W}
