import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = DIMENSIONS
    # Rank 49 decomposition using recursive Strassen
    rank = 49
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Base 2x2 Strassen coefficients
    # u, v, w are length 4 vectors corresponding to indices 00, 01, 10, 11
    # W index mapping for 2x2: C[i,k] -> k*2 + i
    # 00 -> 0, 01 -> 2, 10 -> 1, 11 -> 3
    
    base_u = [
        [1, 0, 0, 1], # M1: A00+A11
        [0, 0, 1, 1], # M2: A10+A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00+A01
        [0, 0, 1, -1],# M6: A10-A00
        [0, 1, 0, -1] # M7: A01-A11
    ]
    
    base_v = [
        [1, 0, 0, 1], # M1: B00+B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01-B11
        [0, 0, 1, -1],# M4: B10-B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00+B01
        [0, 0, 1, 1]  # M7: B10+B11
    ]
    
    base_w = [
        [1, 0, 0, 1], # M1: C00 + C11 (idx 0, 3)
        [0, 1, 0, -1],# M2: C10 - C11 (idx 1, 3)
        [0, 0, 1, 1], # M3: C01 + C11 (idx 2, 3)
        [1, 1, 0, 0], # M4: C00 + C10 (idx 0, 1)
        [-1, 0, 1, 0],# M5: -C00 + C01 (idx 0, 2)
        [0, 0, 0, 1], # M6: C11 (idx 3)
        [1, 0, 0, 0]  # M7: C00 (idx 0)
    ]
    
    # Super-step coefficients for 4x4
    # left_coeffs[s][u][v] is coeff of A_block[u][v] in X_s
    # right_coeffs[s][u][v] is coeff of B_block[u][v] in Y_s
    # out_coeffs[s][p][q] is coeff of Z_s in C_block[p][q]
    
    # s=0: S1 = (A00+A11)(B00+B11) -> C00+=S1, C11+=S1
    left_coeffs = [
        [[1, 0], [0, 1]], # S1
        [[0, 0], [1, 1]], # S2: A10+A11
        [[1, 0], [0, 0]], # S3: A00
        [[0, 0], [0, 1]], # S4: A11
        [[1, 1], [0, 0]], # S5: A00+A01
        [[0, 0], [1, -1]],# S6: A10-A00
        [[0, 1], [0, -1]] # S7: A01-A11
    ]
    
    right_coeffs = [
        [[1, 0], [0, 1]], # S1
        [[1, 0], [0, 0]], # S2
        [[0, 1], [0, -1]],# S3
        [[0, 0], [1, -1]],# S4
        [[0, 0], [0, 1]], # S5
        [[1, 1], [0, 0]], # S6
        [[0, 0], [1, 1]]  # S7
    ]
    
    out_coeffs = [
        [[1, 0], [0, 1]], # S1 -> C00, C11
        [[0, 0], [1, -1]],# S2 -> C10, C11
        [[0, 1], [0, 1]], # S3 -> C01, C11
        [[1, 0], [1, 0]], # S4 -> C00, C10
        [[-1, 1], [0, 0]],# S5 -> C00, C01
        [[0, 0], [0, 1]], # S6 -> C11
        [[1, 0], [0, 0]]  # S7 -> C00
    ]
    
    # Generate the 49 terms
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            u_sub = base_u[t]
            v_sub = base_v[t]
            w_sub = base_w[t]
            
            lc = left_coeffs[s]
            rc = right_coeffs[s]
            oc = out_coeffs[s]
            
            # Compute U global
            for i in range(4):
                for j in range(4):
                    u_val = 0
                    # i = 2*u + r, j = 2*v + c
                    u_block = i // 2
                    r = i % 2
                    v_block = j // 2
                    c = j % 2
                    
                    # X_s[r, c] = sum_{u,v} lc[u][v] A[2u+r, 2v+c]
                    # u_sub index is r*2+c
                    coeff_in_X = u_sub[r * 2 + c]
                    if coeff_in_X != 0:
                        # Coefficient of A[i,j] is coeff_in_X * lc[u_block][v_block]
                        u_val = coeff_in_X * lc[u_block][v_block]
                    
                    U[i * 4 + j][col] = u_val
            
            # Compute V global
            for j in range(4):
                for k in range(4):
                    v_val = 0
                    j_block = j // 2
                    r = j % 2
                    k_block = k // 2
                    c = k % 2
                    
                    coeff_in_Y = v_sub[r * 2 + c]
                    if coeff_in_Y != 0:
                        v_val = coeff_in_Y * rc[j_block][k_block]
                    
                    V[j * 4 + k][col] = v_val
            
            # Compute W global
            for k in range(4):
                for i in range(4):
                    w_val = 0
                    # W index is k*4 + i
                    # C[i, k]
                    p_block = i // 2
                    r = i % 2
                    q_block = k // 2
                    c = k % 2
                    
                    # Z_s[p, q] contributes to C_block[p][q] with oc[p][q]
                    # w_sub index is p*2+q
                    coeff_in_Z = w_sub[p_block * 2 + q_block]
                    if coeff_in_Z != 0:
                        w_val = coeff_in_Z * oc[p_block][q_block]
                    
                    W[k * 4 + i][col] = w_val
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
