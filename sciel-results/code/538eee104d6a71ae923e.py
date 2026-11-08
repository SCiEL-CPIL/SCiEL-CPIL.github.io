import numpy as np
import time

DIMENSIONS = [2, 4, 7]
START_TIME = time.time()
TIME_LIMIT = 140  # safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS  # n=2 (rows of A), m=4 (inner), p=7 (cols of B)
    rank = 50
    U = [[0] * rank for _ in range(n * m)]   # shape (n*m, rank) = (8, 50)
    V = [[0] * rank for _ in range(m * p)]   # shape (m*p, rank) = (28, 50)
    W = [[0] * rank for _ in range(p * n)]   # shape (p*n, rank) = (14, 50)
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        # A[i, j] index in U: i*m + j
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        # B[j, k] index in V: j*p + k
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        # C[i, k] index in W: k*n + i
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a+d)(e+h)
        U[a_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        V[h_idx][col] += 1
        # C00 += M1, C11 += M1
        W[c00_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M2 = (c+d)e
        U[c_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        # C10 += M2, C11 -= M2
        W[c10_idx][col] += 1
        W[c11_idx][col] -= 1
        col += 1
        
        # M3 = a(f-h)
        U[a_idx][col] += 1
        V[f_idx][col] += 1
        V[h_idx][col] -= 1
        # C01 += M3, C11 += M3
        W[c01_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M4 = d(g-e)
        U[d_idx][col] += 1
        V[g_idx][col] += 1
        V[e_idx][col] -= 1
        # C00 += M4, C10 += M4
        W[c00_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1
        
        # M5 = (a+b)h
        U[a_idx][col] += 1
        U[b_idx][col] += 1
        V[h_idx][col] += 1
        # C00 -= M5, C01 += M5
        W[c00_idx][col] -= 1
        W[c01_idx][col] += 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[c_idx][col] += 1
        U[a_idx][col] -= 1
        V[e_idx][col] += 1
        V[f_idx][col] += 1
        # C11 += M6
        W[c11_idx][col] += 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[b_idx][col] += 1
        U[d_idx][col] -= 1
        V[g_idx][col] += 1
        V[h_idx][col] += 1
        # C00 += M7
        W[c00_idx][col] += 1
        col += 1
    
    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)
        
        # P1 = a*e -> C00 += P1
        U[a_idx][col] += 1
        V[e_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1
        
        # P2 = b*g -> C00 += P2
        U[b_idx][col] += 1
        V[g_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1
        
        # P3 = c*e -> C10 += P3
        U[c_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1
        
        # P4 = d*g -> C10 += P4
        U[d_idx][col] += 1
        V[g_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1
    
    # Two j-groups: {0,1} and {2,3}. Each handles 7 output cols as 2+2+2+1.
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        add_schoolbook_221_block(0, j_start, 6)
    
    return {"U": U, "V": V, "W": W}
