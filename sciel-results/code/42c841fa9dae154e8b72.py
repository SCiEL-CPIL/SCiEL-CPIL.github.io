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

    def build_rank_29():
        # Rank 29 construction:
        # 7 Strassen terms for the (i in {0,1}, j in {0,1}, k in {0,1}) block
        # 22 Schoolbook terms for the rest
        
        rank = 29
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        # Helper to add a term
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

        # Strassen terms for 2x2x2 block (i<2, j<2, k<2)
        # Indices: i in {0,1}, j in {0,1}, k in {0,1}
        
        # M1 = (A00 + A11)(B00 + B11)
        u1 = np.zeros((2, 3)); v1 = np.zeros((3, 5)); w1 = np.zeros((5, 2))
        u1[0,0] = 1; u1[1,1] = 1
        v1[0,0] = 1; v1[1,1] = 1
        w1[0,0] = 1; w1[0,1] = 1; w1[1,0] = 1; w1[1,1] = 1
        # Note: Strassen output is C = M1 - M4 - M6 + M7 for C00?
        # Standard Strassen:
        # C00 = M1 - M4 - M6 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M5 - M7
        # We need to assign signs to W correctly.
        # The term is u \otimes v \otimes w.
        # The contribution to C[i,k] is sum_r U[i,j,r] V[j,k,r] W[k,i,r].
        # So W[k,i] should be the coefficient of C[i,k] in that Strassen term.
        
        # Let's define the 7 terms explicitly with correct W signs.
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        # Contributes to C00 (+1), C11 (+1), C01 (0), C10 (0)?
        # Wait, M1 is used in C00 and C11.
        # C00 = M1 - M4 - M6 + M7
        # C11 = M1 - M3 + M5 - M7
        # So M1 has coeff +1 for C00 and +1 for C11.
        
        u1 = np.zeros((2, 3)); v1 = np.zeros((3, 5)); w1 = np.zeros((5, 2))
        u1[0,0] = 1; u1[1,1] = 1
        v1[0,0] = 1; v1[1,1] = 1
        w1[0,0] = 1 # C00
        w1[1,1] = 1 # C11
        add_term(u1, v1, w1)

        # Term 2: M2 = (A10+A11)B00
        # Contributes to C10 (+1)
        u2 = np.zeros((2, 3)); v2 = np.zeros((3, 5)); w2 = np.zeros((5, 2))
        u2[1,0] = 1; u2[1,1] = 1
        v2[0,0] = 1
        w2[0,1] = 1 # C10
        add_term(u2, v2, w2)

        # Term 3: M3 = A00(B10-B11)
        # Contributes to C01 (+1), C11 (-1)
        u3 = np.zeros((2, 3)); v3 = np.zeros((3, 5)); w3 = np.zeros((5, 2))
        u3[0,0] = 1
        v3[1,0] = 1; v3[1,1] = -1
        w3[0,0] = 1 # C01
        w3[1,1] = -1 # C11
        add_term(u3, v3, w3)

        # Term 4: M4 = A11(B01-B00)
        # Contributes to C00 (-1), C10 (+1)
        u4 = np.zeros((2, 3)); v4 = np.zeros((3, 5)); w4 = np.zeros((5, 2))
        u4[1,1] = 1
        v4[0,1] = 1; v4[0,0] = -1
        w4[0,0] = -1 # C00
        w4[0,1] = 1 # C10
        add_term(u4, v4, w4)

        # Term 5: M5 = (A00+A01)B11
        # Contributes to C01 (+1), C11 (+1)
        u5 = np.zeros((2, 3)); v5 = np.zeros((3, 5)); w5 = np.zeros((5, 2))
        u5[0,0] = 1; u5[0,1] = 1
        v5[1,1] = 1
        w5[0,0] = 1 # C01
        w5[1,1] = 1 # C11
        add_term(u5, v5, w5)

        # Term 6: M6 = (A10-A00)(B00+B10)
        # Contributes to C00 (-1)
        u6 = np.zeros((2, 3)); v6 = np.zeros((3, 5)); w6 = np.zeros((5, 2))
        u6[1,0] = 1; u6[0,0] = -1
        v6[0,0] = 1; v6[1,0] = 1
        w6[0,0] = -1 # C00
        add_term(u6, v6, w6)

        # Term 7: M7 = (A01-A11)(B01+B11)
        # Contributes to C00 (+1), C11 (-1)
        u7 = np.zeros((2, 3)); v7 = np.zeros((3, 5)); w7 = np.zeros((5, 2))
        u7[0,1] = 1; u7[1,1] = -1
        v7[0,1] = 1; v7[1,1] = 1
        w7[0,0] = 1 # C00
        w7[1,1] = -1 # C11
        add_term(u7, v7, w7)

        # Add remaining schoolbook terms
        # Skip if j in {0,1} and k in {0,1}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if j < 2 and k < 2:
                        continue
                    
                    us = np.zeros((2, 3)); vs = np.zeros((3, 5)); ws = np.zeros((5, 2))
                    us[i, j] = 1
                    vs[j, k] = 1
                    ws[k, i] = 1
                    add_term(us, vs, ws)

        if col != rank:
            return None
            
        if validate(U, V, W):
            return {"U": U, "V": V, "W": W}
        return None

    result = build_rank_29()
    if result:
        return result

    return get_schoolbook()
