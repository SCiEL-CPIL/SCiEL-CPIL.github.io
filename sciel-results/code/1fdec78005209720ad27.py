import time
import numpy as np
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Build Strassen-based 4x4x4 decomposition (rank 49)
    # and schoolbook 4x4x3 decomposition (rank 48)
    
    # 2x2x2 Strassen basis terms
    s_terms = []
    
    # T1: (a00+a11)(b00+b11) -> c00+=1, c11+=1
    u1 = {(0,0): 1, (1,1): 1}
    v1 = {(0,0): 1, (1,1): 1}
    w1 = {(0,0): 1, (1,1): 1}
    s_terms.append((u1, v1, w1))
    
    # T2: (a10+a11)b00 -> c10+=1, c11-=1
    u2 = {(1,0): 1, (1,1): 1}
    v2 = {(0,0): 1}
    w2 = {(1,0): 1, (1,1): -1}
    s_terms.append((u2, v2, w2))
    
    # T3: a00(b01-b11) -> c01+=1, c11+=1
    u3 = {(0,0): 1}
    v3 = {(0,1): 1, (1,1): -1}
    w3 = {(0,1): 1, (1,1): 1}
    s_terms.append((u3, v3, w3))
    
    # T4: a11(b10-b00) -> c00+=1, c10+=1
    u4 = {(1,1): 1}
    v4 = {(1,0): 1, (0,0): -1}
    w4 = {(0,0): 1, (1,0): 1}
    s_terms.append((u4, v4, w4))
    
    # T5: (a00+a01)b11 -> c00-=1, c01+=1
    u5 = {(0,0): 1, (0,1): 1}
    v5 = {(1,1): 1}
    w5 = {(0,0): -1, (0,1): 1}
    s_terms.append((u5, v5, w5))
    
    # T6: (a10-a00)(b00+b01) -> c11+=1
    u6 = {(1,0): 1, (0,0): -1}
    v6 = {(0,0): 1, (0,1): 1}
    w6 = {(1,1): 1}
    s_terms.append((u6, v6, w6))
    
    # T7: (a01-a11)(b10+b11) -> c00+=1
    u7 = {(0,1): 1, (1,1): -1}
    v7 = {(1,0): 1, (1,1): 1}
    w7 = {(0,0): 1}
    s_terms.append((u7, v7, w7))
    
    all_terms = []
    
    # M1: (A00+A11)(B00+B11)
    for u, v, w in s_terms:
        u_new = {}
        v_new = {}
        w_new = {}
        for (li, lj), coeff in u.items():
            u_new[(li, lj)] = coeff
            u_new[(li+2, lj+2)] = coeff
        for (li, lj), coeff in v.items():
            v_new[(li, lj)] = coeff
            v_new[(li+2, lj+2)] = coeff
        for (li, lj), coeff in w.items():
            w_new[(li, lj)] = coeff
            w_new[(li+2, lj+2)] = coeff
        all_terms.append((u_new, v_new, w_new))
    
    # M2: (A10+A11)B00
    for u, v, w in s_terms:
        u_new = {}
        v_new = {}
        w_new = {}
        for (li, lj), coeff in u.items():
            u_new[(li+2, lj)] = coeff
            u_new[(li+2, lj+2)] = coeff
        for (li, lj), coeff in v.items():
            v_new[(li, lj)] = coeff
        for (li, lj), coeff in w.items():
            w_new[(li+2, lj)] = coeff
            w_new[(li+2, lj+2)] = -coeff
        all_terms.append((u_new, v_new, w_new))
    
    # M3: A00(B01-B11)
    for u, v, w in s_terms:
        u_new = {}
        v_new = {}
        w_new = {}
        for (li, lj), coeff in u.items():
            u_new[(li, lj)] = coeff
        for (li, lj), coeff in v.items():
            v_new[(li, lj+2)] = coeff
            v_new[(li+2, lj+2)] = -coeff
        for (li, lj), coeff in w.items():
            w_new[(li, lj+2)] = coeff
            w_new[(li+2, lj+2)] = coeff
        all_terms.append((u_new, v_new, w_new))
    
    # M4: A11(B10-B00)
    for u, v, w in s_terms:
        u_new = {}
        v_new = {}
        w_new = {}
        for (li, lj), coeff in u.items():
            u_new[(li+2, lj+2)] = coeff
        for (li, lj), coeff in v.items():
            v_new[(li+2, lj)] = coeff
            v_new[(li, lj)] = -coeff
        for (li, lj), coeff in w.items():
            w_new[(li, lj)] = coeff
            w_new[(li+2, lj)] = coeff
        all_terms.append((u_new, v_new, w_new))
    
    # M5: (A00+A01)B11
    for u, v, w in s_terms:
        u_new = {}
        v_new = {}
        w_new = {}
        for (li, lj), coeff in u.items():
            u_new[(li, lj)] = coeff
            u_new[(li, lj+2)] = coeff
        for (li, lj), coeff in v.items():
            v_new[(li+2, lj+2)] = coeff
        for (li, lj), coeff in w.items():
            w_new[(li, lj)] = -coeff
            w_new[(li, lj+2)] = coeff
        all_terms.append((u_new, v_new, w_new))
    
    # M6: (A10-A00)(B00+B01)
    for u, v, w in s_terms:
        u_new = {}
        v_new = {}
        w_new = {}
        for (li, lj), coeff in u.items():
            u_new[(li+2, lj)] = coeff
            u_new[(li, lj)] = -coeff
        for (li, lj), coeff in v.items():
            v_new[(li, lj)] = coeff
            v_new[(li, lj+2)] = coeff
        for (li, lj), coeff in w.items():
            w_new[(li+2, lj+2)] = coeff
        all_terms.append((u_new, v_new, w_new))
    
    # M7: (A01-A11)(B10+B11)
    for u, v, w in s_terms:
        u_new = {}
        v_new = {}
        w_new = {}
        for (li, lj), coeff in u.items():
            u_new[(li, lj+2)] = coeff
            u_new[(li+2, lj+2)] = -coeff
        for (li, lj), coeff in v.items():
            v_new[(li+2, lj)] = coeff
            v_new[(li+2, lj+2)] = coeff
        for (li, lj), coeff in w.items():
            w_new[(li, lj)] = coeff
        all_terms.append((u_new, v_new, w_new))
    
    # Schoolbook for 4x4x3 (columns 4,5,6 of B)
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                u = {(i, j): 1}
                v = {(j, k): 1}
                w = {(k, i): 1}
                all_terms.append((u, v, w))
    
    rank = len(all_terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(all_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][c] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][c] = coeff
        for (k, i), coeff in w_dict.items():
            W[k * n + i][c] = coeff
    
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    valid = True
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for l in range(m):
                    s = 0.0
                    for c in range(rank):
                        s += W_arr[k * n + i, c] * U_arr[i * m + j, c] * V_arr[l * p + k, c]
                    expected = 1.0 if j == l else 0.0
                    if abs(s - expected) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
