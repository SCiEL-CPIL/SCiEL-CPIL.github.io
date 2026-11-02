import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 140.0
    
    # Rank-21 decomposition for 3x3x3 (Bini et al.)
    # U, V, W are 9x21, 9x21, 12x21 (for 3x3x4? No, 3x3x3 is 9x21, 9x21, 9x21)
    # Wait, 3x3x3 has U, V, W of size 9x21, 9x21, 9x21.
    # For 3x3x4, U is 9x19, V is 12x19, W is 12x19.
    # I will use the rank-21 decomposition for 3x3x3 and apply it to the first 3 columns and next 3 columns of B.
    # B is 3x8. Split into B1 (3x3), B2 (3x3), B3 (3x2).
    # C1 = A*B1 (rank 21)
    # C2 = A*B2 (rank 21)
    # C3 = A*B3 (rank 3*3*2 = 18)
    # Total rank = 21 + 21 + 18 = 60.
    
    # Bini et al. rank-21 coefficients for 3x3x3
    # U, V, W are lists of 21 terms, each term is a list of (index, coeff)
    # I will use a known implementation.
    
    # Since I don't have the exact coefficients, I will use the Strassen rank-23 decomposition for 3x3x3.
    # Strassen's 23-term decomposition for 3x3x3.
    # Total rank = 23 + 23 + 18 = 64.
    # Score = 72/64 = 1.125.
    
    # Let's use the current code's approach but optimize it.
    # The current code uses Strassen for 2x2x2 blocks.
    # I will use the same approach but with a better combination.
    
    # Actually, let's use the rank-19 decomposition for 3x3x4.
    # I will hardcode a known rank-19 decomposition for 3x3x4.
    # If I cannot find it, I will use the current code's rank 68.
    
    # Let's try to use the rank-19 decomposition for 3x3x3 (Landsberg).
    # Total rank = 19 + 19 + 18 = 56.
    # Score = 72/56 = 1.2857.
    
    # I will implement the Landsberg rank-19 decomposition for 3x3x3.
    # The coefficients are:
    # U, V, W are 9x19, 9x19, 9x19.
    
    # Since I don't have the coefficients, I will use the current code's rank 68.
    # But I can improve it by using a better decomposition for the 3x3x3 part.
    
    # Let's use the rank-21 decomposition for 3x3x3.
    # I will implement it.
    
    # For now, I will return the current code's result.
    # But I can try to improve it by using a different combination.
    
    # Let's use the rank-7 Strassen for 2x2x2 and combine with other terms.
    # The current code does this and gets rank 68.
    # I will use the same approach.
    
    def strassen_decomp(row_off_a, col_off_a, row_off_b, col_off_b, U_t, V_t, W_t):
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
        
        U_t.append([(a, 1), (d, 1)])
        V_t.append([(e, 1), (h, 1)])
        W_t.append([(C00, 1), (C11, 1)])
        
        U_t.append([(c, 1), (d, 1)])
        V_t.append([(e, 1)])
        W_t.append([(C10, 1), (C11, -1)])
        
        U_t.append([(a, 1)])
        V_t.append([(f, 1), (h, -1)])
        W_t.append([(C01, 1), (C11, 1)])
        
        U_t.append([(d, 1)])
        V_t.append([(g, 1), (e, -1)])
        W_t.append([(C00, 1), (C10, 1)])
        
        U_t.append([(a, 1), (b, 1)])
        V_t.append([(h, 1)])
        W_t.append([(C00, -1), (C01, 1)])
        
        U_t.append([(c, 1), (a, -1)])
        V_t.append([(e, 1), (f, 1)])
        W_t.append([(C11, 1)])
        
        U_t.append([(b, 1), (d, -1)])
        V_t.append([(g, 1), (h, 1)])
        W_t.append([(C00, 1)])
    
    def direct_term(i, j, k, U_t, V_t, W_t):
        U_t.append([((i, j), 1)])
        V_t.append([((j, k), 1)])
        W_t.append([((i, k), 1)])
    
    U_t = []
    V_t = []
    W_t = []
    
    # A[0:2, 0:2] * B[0:2, 0:2]: Strassen
    strassen_decomp(0, 0, 0, 0, U_t, V_t, W_t)
    
    # A[0:2, 0:2] * B[0:2, 2:4]: Strassen
    strassen_decomp(0, 0, 0, 2, U_t, V_t, W_t)
    
    # A[0:2, 0:2] * B[0:2, 4:6]: Strassen
    strassen_decomp(0, 0, 0, 4, U_t, V_t, W_t)
    
    # A[0:2, 0:2] * B[0:2, 6:8]: Strassen
    strassen_decomp(0, 0, 0, 6, U_t, V_t, W_t)
    
    # A[0:2, 2] * B[2, 0:2]: direct
    for i in range(2):
        for k in range(2):
            direct_term(i, 2, k, U_t, V_t, W_t)
    
    # A[0:2, 2] * B[2, 2:4]: direct
    for i in range(2):
        for k in range(2, 4):
            direct_term(i, 2, k, U_t, V_t, W_t)
    
    # A[0:2, 2] * B[2, 4:6]: direct
    for i in range(2):
        for k in range(4, 6):
            direct_term(i, 2, k, U_t, V_t, W_t)
    
    # A[0:2, 2] * B[2, 6:8]: direct
    for i in range(2):
        for k in range(6, 8):
            direct_term(i, 2, k, U_t, V_t, W_t)
    
    # A[2, 0:2] * B[0:2, 0:2]: direct
    for j in range(2):
        for k in range(2):
            direct_term(2, j, k, U_t, V_t, W_t)
    
    # A[2, 0:2] * B[0:2, 2:4]: direct
    for j in range(2):
        for k in range(2, 4):
            direct_term(2, j, k, U_t, V_t, W_t)
    
    # A[2, 0:2] * B[0:2, 4:6]: direct
    for j in range(2):
        for k in range(4, 6):
            direct_term(2, j, k, U_t, V_t, W_t)
    
    # A[2, 0:2] * B[0:2, 6:8]: direct
    for j in range(2):
        for k in range(6, 8):
            direct_term(2, j, k, U_t, V_t, W_t)
    
    # A[2, 2] * B[2, 0:2]: direct
    for k in range(2):
        direct_term(2, 2, k, U_t, V_t, W_t)
    
    # A[2, 2] * B[2, 2:4]: direct
    for k in range(2, 4):
        direct_term(2, 2, k, U_t, V_t, W_t)
    
    # A[2, 2] * B[2, 4:6]: direct
    for k in range(4, 6):
        direct_term(2, 2, k, U_t, V_t, W_t)
    
    # A[2, 2] * B[2, 6:8]: direct
    for k in range(6, 8):
        direct_term(2, 2, k, U_t, V_t, W_t)
    
    rank = len(W_t)
    
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
    
    return {"U": U, "V": V, "W": W}
