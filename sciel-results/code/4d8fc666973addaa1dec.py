import time
import numpy as np
import random
from math import gcd
from functools import reduce

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Schoolbook fallback
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def validate(U, V, W):
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_arr[i*m+j]
                    v_val = V_arr[j*p+k]
                    w_val = W_arr[k*n+i]
                    s = np.dot(u_val, v_val * w_val)
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def build_rank_17():
        rank = 17
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        def add_term(u_vals, v_vals, w_vals):
            nonlocal col
            for i in range(n):
                for j in range(m):
                    if u_vals[i, j] != 0:
                        U[i*m + j][col] = u_vals[i, j]
            for j in range(m):
                for k in range(p):
                    if v_vals[j, k] != 0:
                        V[j*p + k][col] = v_vals[j, k]
            for k in range(p):
                for i in range(n):
                    if w_vals[k, i] != 0:
                        W[k*n + i][col] = w_vals[k, i]
            col += 1

        # Helper to add Strassen terms for a 2x2x2 block
        # i in {0,1}, j in {0,1}, k in {k_offset, k_offset+1}
        def add_strassen_block(k_offset):
            nonlocal col
            # 7 Strassen terms
            # M1 = (A00 + A11)(B00 + B11)
            u1 = np.zeros((2, 3)); v1 = np.zeros((3, 5)); w1 = np.zeros((5, 2))
            u1[0,0] = 1; u1[1,1] = 1
            v1[0, k_offset] = 1; v1[1, k_offset+1] = 1
            w1[k_offset,0] = 1; w1[k_offset+1,1] = 1
            add_term(u1, v1, w1)

            # M2 = (A10 + A11)B00
            u2 = np.zeros((2, 3)); v2 = np.zeros((3, 5)); w2 = np.zeros((5, 2))
            u2[1,0] = 1; u2[1,1] = 1
            v2[0, k_offset] = 1
            w2[k_offset,1] = 1
            add_term(u2, v2, w2)

            # M3 = A00(B10 - B11)
            u3 = np.zeros((2, 3)); v3 = np.zeros((3, 5)); w3 = np.zeros((5, 2))
            u3[0,0] = 1
            v3[1, k_offset] = 1; v3[1, k_offset+1] = -1
            w3[k_offset,0] = 1; w3[k_offset+1,1] = -1
            add_term(u3, v3, w3)

            # M4 = A11(B01 - B00)
            u4 = np.zeros((2, 3)); v4 = np.zeros((3, 5)); w4 = np.zeros((5, 2))
            u4[1,1] = 1
            v4[0, k_offset+1] = 1; v4[0, k_offset] = -1
            w4[k_offset,0] = -1; w4[k_offset,1] = 1
            add_term(u4, v4, w4)

            # M5 = (A00 + A01)B11
            u5 = np.zeros((2, 3)); v5 = np.zeros((3, 5)); w5 = np.zeros((5, 2))
            u5[0,0] = 1; u5[0,1] = 1
            v5[1, k_offset+1] = 1
            w5[k_offset,0] = 1; w5[k_offset+1,1] = 1
            add_term(u5, v5, w5)

            # M6 = (A10 - A00)(B00 + B10)
            u6 = np.zeros((2, 3)); v6 = np.zeros((3, 5)); w6 = np.zeros((5, 2))
            u6[1,0] = 1; u6[0,0] = -1
            v6[0, k_offset] = 1; v6[1, k_offset] = 1
            w6[k_offset,0] = -1
            add_term(u6, v6, w6)

            # M7 = (A01 - A11)(B01 + B11)
            u7 = np.zeros((2, 3)); v7 = np.zeros((3, 5)); w7 = np.zeros((5, 2))
            u7[0,1] = 1; u7[1,1] = -1
            v7[0, k_offset+1] = 1; v7[1, k_offset+1] = 1
            w7[k_offset,0] = 1; w7[k_offset+1,1] = -1
            add_term(u7, v7, w7)

        # Add Strassen block for k in {0, 1}
        add_strassen_block(0)
        
        # Add j=2 term for k in {0, 1}
        u_j2_01 = np.zeros((2, 3)); v_j2_01 = np.zeros((3, 5)); w_j2_01 = np.zeros((5, 2))
        u_j2_01[0,2] = 1; u_j2_01[1,2] = 1
        v_j2_01[2,0] = 1; v_j2_01[2,1] = 1
        w_j2_01[0,0] = 1; w_j2_01[0,1] = 1; w_j2_01[1,0] = 1; w_j2_01[1,1] = 1
        add_term(u_j2_01, v_j2_01, w_j2_01)

        # Add Strassen block for k in {3, 4}
        add_strassen_block(3)
        
        # Add j=2 term for k in {3, 4}
        u_j2_34 = np.zeros((2, 3)); v_j2_34 = np.zeros((3, 5)); w_j2_34 = np.zeros((5, 2))
        u_j2_34[0,2] = 1; u_j2_34[1,2] = 1
        v_j2_34[2,3] = 1; v_j2_34[2,4] = 1
        w_j2_34[3,0] = 1; w_j2_34[3,1] = 1; w_j2_34[4,0] = 1; w_j2_34[4,1] = 1
        add_term(u_j2_34, v_j2_34, w_j2_34)

        # Add k=2 term (covers all j)
        u_k2 = np.zeros((2, 3)); v_k2 = np.zeros((3, 5)); w_k2 = np.zeros((5, 2))
        u_k2[0,0] = 1; u_k2[0,1] = 1; u_k2[0,2] = 1
        u_k2[1,0] = 1; u_k2[1,1] = 1; u_k2[1,2] = 1
        v_k2[0,2] = 1; v_k2[1,2] = 1; v_k2[2,2] = 1
        w_k2[2,0] = 1; w_k2[2,1] = 1
        add_term(u_k2, v_k2, w_k2)

        if col != rank:
            return None
            
        if validate(U, V, W):
            return {"U": U, "V": V, "W": W}
        return None

    result = build_rank_17()
    if result:
        return result

    return get_schoolbook()
