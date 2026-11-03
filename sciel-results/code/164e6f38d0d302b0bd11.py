"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses Strassen's rank-7 decomposition for 2x2x2 blocks combined with
direct terms for remaining dimensions, targeting rank 68.
"""

import time
import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 130.0  # safety margin before 150s kill
    
    def strassen_block(row_off_a, col_off_a, row_off_b, col_off_b):
        """Return list of (U_terms, V_terms, W_terms) for one 2x2*2x2 Strassen block.
        
        Strassen:
        m1 = (a+d)(e+h)
        m2 = (c+d)*e
        m3 = a*(f-h)
        m4 = d*(g-e)
        m5 = (a+b)*h
        m6 = (c-a)*(e+f)
        m7 = (b-d)*(g+h)
        
        C[0,0] = m1 + m4 - m5 + m7
        C[0,1] = m3 + m5
        C[1,0] = m2 + m4
        C[1,1] = m1 - m2 + m3 + m6
        """
        a = (row_off_a, col_off_a)
        b = (row_off_a, col_off_a + 1)
        c = (row_off_a + 1, col_off_a)
        d = (row_off_a + 1, col_off_a + 1)
        e = (row_off_b, col_off_b)
        f = (row_off_b, col_off_b + 1)
        g = (row_off_b + 1, col_off_b)
        h = (row_off_b + 1, col_off_b + 1)
        
        C00 = (row_off_a, col_off_b)
        C01 = (row_off_a, col_off_b + 1)
        C10 = (row_off_a + 1, col_off_b)
        C11 = (row_off_a + 1, col_off_b + 1)
        
        blocks = []
        
        # m1 = (a+d)(e+h)
        blocks.append(([(a, 1), (d, 1)], [(e, 1), (h, 1)], [(C00, 1), (C11, 1)]))
        
        # m2 = (c+d)*e
        blocks.append(([(c, 1), (d, 1)], [(e, 1)], [(C10, 1), (C11, -1)]))
        
        # m3 = a*(f-h)
        blocks.append(([(a, 1)], [(f, 1), (h, -1)], [(C01, 1), (C11, 1)]))
        
        # m4 = d*(g-e)
        blocks.append(([(d, 1)], [(g, 1), (e, -1)], [(C00, 1), (C10, 1)]))
        
        # m5 = (a+b)*h
        blocks.append(([(a, 1), (b, 1)], [(h, 1)], [(C00, -1), (C01, 1)]))
        
        # m6 = (c-a)*(e+f)
        blocks.append(([(c, 1), (a, -1)], [(e, 1), (f, 1)], [(C11, 1)]))
        
        # m7 = (b-d)*(g+h)
        blocks.append(([(b, 1), (d, -1)], [(g, 1), (h, 1)], [(C00, 1)]))
        
        return blocks
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    # Four Strassen blocks: A[0:2, 0:2] * B[0:2, k:k+2] for k=0,2,4,6
    for col_off_b in [0, 2, 4, 6]:
        blocks = strassen_block(0, 0, 0, col_off_b)
        for u, v, w in blocks:
            U_terms.append(u)
            V_terms.append(v)
            W_terms.append(w)
    
    # Direct terms: A[0:2, 2] * B[2, 0:8]
    for i in range(2):
        for k in range(8):
            U_terms.append([((i, 2), 1)])
            V_terms.append([((2, k), 1)])
            W_terms.append([((i, k), 1)])
    
    # Direct terms: A[2, 0:2] * B[0:2, 0:8]
    for j in range(2):
        for k in range(8):
            U_terms.append([((2, j), 1)])
            V_terms.append([((j, k), 1)])
            W_terms.append([((2, k), 1)])
    
    # Direct terms: A[2, 2] * B[2, 0:8]
    for k in range(8):
        U_terms.append([((2, 2), 1)])
        V_terms.append([((2, k), 1)])
        W_terms.append([((2, k), 1)])
    
    rank = len(W_terms)
    
    # Build factor matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for idx in range(rank):
        for (i, j), coeff in U_terms[idx]:
            U[i * m + j][idx] = coeff
        for (j, k), coeff in V_terms[idx]:
            V[j * p + k][idx] = coeff
        for (i, k), coeff in W_terms[idx]:
            W[k * n + i][idx] = coeff
    
    # Verify correctness
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    def verify_decomp(Ua, Va, Wa):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ua[i*m+j], Va[j*p+k])
                    s = s * Wa[k*n+i]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify_decomp(U_arr, V_arr, W_arr):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
