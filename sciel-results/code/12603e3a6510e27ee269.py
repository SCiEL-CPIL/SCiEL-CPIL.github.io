import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    deadline = time.time() + 110
    
    def strassen_2x2x2():
        U = [
            [1, 0, 0, 1],  # m1
            [0, 0, 1, 1],  # m2
            [1, 0, 0, 0],  # m3
            [0, 0, 0, 1],  # m4
            [1, 1, 0, 0],  # m5
            [0, -1, 1, 0], # m6
            [0, 1, 0, -1], # m7
        ]
        V = [
            [1, 0, 0, 1],  # m1
            [1, 0, 0, 0],  # m2
            [0, 1, 0, -1], # m3
            [0, 0, 1, -1], # m4
            [0, 0, 0, 1],  # m5
            [1, 1, 0, 0],  # m6
            [0, 0, 1, 1],  # m7
        ]
        W = [
            [1, 0, 0, 1],  # m1: C00 + C11
            [0, 0, 1, 0],  # m2: C10
            [0, 1, 0, 0],  # m3: C01
            [0, 0, 1, 1],  # m4: C10 + C11
            [1, 1, 0, 0],  # m5: C00 + C01
            [0, 0, 0, 1],  # m6: C11
            [1, 0, 0, 1],  # m7: C00 + C11
        ]
        return U, V, W
    
    block_A_coeffs = [
        [1, 0, 0, 1],  # M1: A11 + A22
        [0, 0, 1, 1],  # M2: A21 + A22
        [1, 0, 0, 0],  # M3: A11
        [0, 0, 0, 1],  # M4: A22
        [1, 1, 0, 0],  # M5: A11 + A12
        [0, 0, 1, -1], # M6: A21 - A11
        [0, 1, 0, -1], # M7: A12 - A22
    ]
    block_B_coeffs = [
        [1, 0, 0, 1],  # M1: B11 + B22
        [1, 0, 0, 0],  # M2: B11
        [0, 1, 0, -1], # M3: B12 - B22
        [0, 0, 1, -1], # M4: B21 - B11
        [0, 0, 0, 1],  # M5: B22
        [1, 1, 0, 0],  # M6: B11 + B12
        [0, 0, 1, 1],  # M7: B21 + B22
    ]
    block_C_coeffs = [
        [1, 0, 0, 1],  # M1: C11 + C22
        [0, 0, 1, 0],  # M2: C21
        [0, 1, 0, 0],  # M3: C12
        [0, 0, 1, 1],  # M4: C21 + C22
        [1, 1, 0, 0],  # M5: C11 + C12
        [0, 0, 0, 1],  # M6: C22
        [1, 0, 0, 1],  # M7: C11 + C22
    ]
    
    U_2x2, V_2x2, W_2x2 = strassen_2x2x2()
    
    U_4x4 = []
    V_4x4 = []
    W_4x4 = []
    
    for k in range(7):
        for r in range(7):
            u_vec = [0] * 16
            for b in range(4):
                if block_A_coeffs[k][b] != 0:
                    coeff = block_A_coeffs[k][b]
                    base = b * 4
                    for w in range(4):
                        u_vec[base + w] += coeff * U_2x2[w][r]
            
            v_vec = [0] * 16
            for b in range(4):
                if block_B_coeffs[k][b] != 0:
                    coeff = block_B_coeffs[k][b]
                    base = b * 4
                    for w in range(4):
                        v_vec[base + w] += coeff * V_2x2[w][r]
            
            w_vec = [0] * 16
            for b in range(4):
                if block_C_coeffs[k][b] != 0:
                    coeff = block_C_coeffs[k][b]
                    base = b * 4
                    for w in range(4):
                        w_vec[base + w] += coeff * W_2x2[w][r]
            
            U_4x4.append(u_vec)
            V_4x4.append(v_vec)
            W_4x4.append(w_vec)
    
    def verify(U, V, W, n, m, p):
        rank = len(U[0]) if U else 0
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] 
                             for r in range(rank))
                    if val != 1:
                        return False
        return True
    
    U_full = []
    V_full = []
    W_full = []
    
    # Part 1: A*B1 (rank 49)
    for r in range(49):
        u = U_4x4[r][:]
        v = [0] * 36
        for j in range(4):
            for k in range(4):
                v[j*9+k] = V_4x4[r][j*4+k]
        w = [0] * 36
        for i in range(4):
            for k in range(4):
                w[k*4+i] = W_4x4[r][k*4+i]
        U_full.append(u)
        V_full.append(v)
        W_full.append(w)
    
    # Part 2: A*B2 (rank 49)
    for r in range(49):
        u = U_4x4[r][:]
        v = [0] * 36
        for j in range(4):
            for k in range(4):
                v[j*9+(4+k)] = V_4x4[r][j*4+k]
        w = [0] * 36
        for i in range(4):
            for k in range(4):
                w[(4+k)*4+i] = W_4x4[r][k*4+i]
        U_full.append(u)
        V_full.append(v)
        W_full.append(w)
    
    # Part 3: A*B3 (rank 4, schoolbook)
    for i in range(4):
        for j in range(4):
            u = [0] * 16
            u[i*4+j] = 1
            v = [0] * 36
            v[j*9+8] = 1
            w = [0] * 36
            w[8*4+i] = 1
            U_full.append(u)
            V_full.append(v)
            W_full.append(w)
    
    rank = len(U_full[0])
    
    if not verify(U_full, V_full, W_full, n, m, p):
        rank_school = n * m * p
        U_school = [[0] * rank_school for _ in range(n * m)]
        V_school = [[0] * rank_school for _ in range(m * p)]
        W_school = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j][col] = 1
                    V_school[j*p+k][col] = 1
                    W_school[k*n+i][col] = 1
                    col += 1
        return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U_full, "V": V_full, "W": W_full}
