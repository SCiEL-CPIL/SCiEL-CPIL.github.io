"""Combined approach: Strassen-based block decomposition for 3x3 * 3x8 matrix multiplication.
Uses the structural idea of the Strassen block decomposition (from inspiration 1)
combined with a verification and fallback mechanism from the current program.

The strategy splits the 3x3 * 3x8 multiplication into 2x2 * 2x2 Strassen blocks
and direct terms, achieving rank 68 vs the naive rank 72.
"""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Build Strassen-based decomposition
    U_t = []
    V_t = []
    W_t = []
    
    def strassen_decomp(row_off_a, col_off_a, row_off_b, col_off_b):
        """Add Strassen decomposition for 2x2*2x2 block at given offsets.
        
        Correct Strassen:
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
        
        # m1 = (a+d)(e+h) -> C00 += 1, C11 += 1
        U_t.append([(a, 1), (d, 1)])
        V_t.append([(e, 1), (h, 1)])
        W_t.append([(C00, 1), (C11, 1)])
        
        # m2 = (c+d)*e -> C10 += 1, C11 -= 1
        U_t.append([(c, 1), (d, 1)])
        V_t.append([(e, 1)])
        W_t.append([(C10, 1), (C11, -1)])
        
        # m3 = a*(f-h) -> C01 += 1, C11 += 1
        U_t.append([(a, 1)])
        V_t.append([(f, 1), (h, -1)])
        W_t.append([(C01, 1), (C11, 1)])
        
        # m4 = d*(g-e) -> C00 += 1, C10 += 1
        U_t.append([(d, 1)])
        V_t.append([(g, 1), (e, -1)])
        W_t.append([(C00, 1), (C10, 1)])
        
        # m5 = (a+b)*h -> C00 -= 1, C01 += 1
        U_t.append([(a, 1), (b, 1)])
        V_t.append([(h, 1)])
        W_t.append([(C00, -1), (C01, 1)])
        
        # m6 = (c-a)*(e+f) -> C11 += 1
        U_t.append([(c, 1), (a, -1)])
        V_t.append([(e, 1), (f, 1)])
        W_t.append([(C11, 1)])
        
        # m7 = (b-d)*(g+h) -> C00 += 1
        U_t.append([(b, 1), (d, -1)])
        V_t.append([(g, 1), (h, 1)])
        W_t.append([(C00, 1)])
    
    def direct_term(i, j, k):
        """Add a direct term A[i,j]*B[j,k] to C[i,k]."""
        U_t.append([((i, j), 1)])
        V_t.append([((j, k), 1)])
        W_t.append([((i, k), 1)])
    
    # A[0:2, 0:2] * B[0:2, 0:2]: Strassen
    strassen_decomp(0, 0, 0, 0)
    
    # A[0:2, 0:2] * B[0:2, 2:4]: Strassen
    strassen_decomp(0, 0, 0, 2)
    
    # A[0:2, 0:2] * B[0:2, 4:6]: Strassen
    strassen_decomp(0, 0, 0, 4)
    
    # A[0:2, 0:2] * B[0:2, 6:8]: Strassen
    strassen_decomp(0, 0, 0, 6)
    
    # A[0:2, 2] * B[2, 0:2]: direct
    for i in range(2):
        for k in range(2):
            direct_term(i, 2, k)
    
    # A[0:2, 2] * B[2, 2:4]: direct
    for i in range(2):
        for k in range(2, 4):
            direct_term(i, 2, k)
    
    # A[0:2, 2] * B[2, 4:6]: direct
    for i in range(2):
        for k in range(4, 6):
            direct_term(i, 2, k)
    
    # A[0:2, 2] * B[2, 6:8]: direct
    for i in range(2):
        for k in range(6, 8):
            direct_term(i, 2, k)
    
    # A[2, 0:2] * B[0:2, 0:2]: direct
    for j in range(2):
        for k in range(2):
            direct_term(2, j, k)
    
    # A[2, 0:2] * B[0:2, 2:4]: direct
    for j in range(2):
        for k in range(2, 4):
            direct_term(2, j, k)
    
    # A[2, 0:2] * B[0:2, 4:6]: direct
    for j in range(2):
        for k in range(4, 6):
            direct_term(2, j, k)
    
    # A[2, 0:2] * B[0:2, 6:8]: direct
    for j in range(2):
        for k in range(6, 8):
            direct_term(2, j, k)
    
    # A[2, 2] * B[2, 0:2]: direct
    for k in range(2):
        direct_term(2, 2, k)
    
    # A[2, 2] * B[2, 2:4]: direct
    for k in range(2, 4):
        direct_term(2, 2, k)
    
    # A[2, 2] * B[2, 4:6]: direct
    for k in range(4, 6):
        direct_term(2, 2, k)
    
    # A[2, 2] * B[2, 6:8]: direct
    for k in range(6, 8):
        direct_term(2, 2, k)
    
    rank = len(W_t)
    
    # Build the factor matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for idx in range(rank):
        for (i, j), coeff in U_t[idx]:
            U[i * m + j][idx] = coeff
        for (j, k), coeff in V_t[idx]:
            V[j * p + k][idx] = coeff
        for (i, k), coeff in W_t[idx]:
            W[k * n + i][idx] = coeff
    
    # Verify correctness
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    if _verify_np(n, m, p, U_arr, V_arr, W_arr):
        return _to_lists(U_arr, V_arr, W_arr)
    
    # Fallback to schoolbook if verification fails
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

def _verify(n, m, p, U, V, W):
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(r))
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _verify_np(n, m, p, U, V, W):
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _to_lists(U, V, W):
    return [[float(x) for x in row] for row in U.tolist()], \
           [[float(x) for x in row] for row in V.tolist()], \
           [[float(x) for x in row] for row in W.tolist()]
# EVOLVE-BLOCK-END
