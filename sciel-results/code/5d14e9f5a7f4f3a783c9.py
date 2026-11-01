import time
import random
import math

def construct():
    n, m, p = 2, 2, 11
    start_time = time.time()
    deadline = start_time + 110
    
    # Rank calculation
    # 5 pairs of columns for Strassen (rank 7 each)
    # 1 single column for schoolbook (rank 4)
    rank = 5 * 7 + 4
    total_rank = rank
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Helper to add a rank-1 term
    # u_vec: dict mapping (i,j) to coeff
    # v_vec: dict mapping (j,k) to coeff
    # w_vec: dict mapping (i,k) to coeff
    # Note: W index is d = k*n + i
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col_idx
        for (i, j), u_val in u_coeffs.items():
            if u_val != 0:
                U[i * m + j][col_idx] += u_val
        for (j, k), v_val in v_coeffs.items():
            if v_val != 0:
                V[j * p + k][col_idx] += v_val
        for (i, k), w_val in w_coeffs.items():
            if w_val != 0:
                W[k * n + i][col_idx] += w_val
        col_idx += 1

    # Strassen terms for a 2x2 block of B columns (k_a, k_b)
    # A is 2x2, B_block is 2x2
    # P1 = A00 * (B00 - B11)
    # P2 = (A00 + A11) * B11
    # P3 = (A10 - A00) * B00
    # P4 = A11 * (B10 - B00)
    # P5 = (A00 + A10) * B01
    # P6 = (A11 - A01) * (B01 + B11)
    # P7 = (A01 - A11) * (B00 + B10)
    
    # Output contributions:
    # C00 = P1 + P2 - P4 + P7
    # C01 = P3 + P4
    # C10 = P1 + P2
    # C11 = P3 + P6 - P1 + P5
    
    # We map the local 2x2 indices (0,1) to global k indices (k_a, k_b)
    # Local B[0,0] -> Global B[0, k_a]
    # Local B[0,1] -> Global B[0, k_b]
    # Local B[1,0] -> Global B[1, k_a]
    # Local B[1,1] -> Global B[1, k_b]
    
    def strassen_block(k_a, k_b):
        # P1
        u1 = {(0,0): 1}
        v1 = {(0, k_a): 1, (1, k_b): -1}
        w1 = {(0, k_a): 1, (1, k_a): 1} # C00, C10
        add_term(u1, v1, w1)
        
        # P2
        u2 = {(0,0): 1, (1,1): 1}
        v2 = {(1, k_b): 1}
        w2 = {(0, k_a): 1, (1, k_a): 1} # C00, C10
        add_term(u2, v2, w2)
        
        # P3
        u3 = {(1,0): 1, (0,0): -1}
        v3 = {(0, k_a): 1}
        w3 = {(0, k_b): 1, (1, k_b): 1} # C01, C11
        add_term(u3, v3, w3)
        
        # P4
        u4 = {(1,1): 1}
        v4 = {(1, k_a): 1, (0, k_a): -1}
        w4 = {(0, k_b): 1, (0, k_a): -1} # C01, C00
        add_term(u4, v4, w4)
        
        # P5
        u5 = {(0,0): 1, (1,0): 1}
        v5 = {(0, k_b): 1}
        w5 = {(1, k_b): 1} # C11
        add_term(u5, v5, w5)
        
        # P6
        u6 = {(1,1): 1, (0,1): -1}
        v6 = {(0, k_b): 1, (1, k_b): 1}
        w6 = {(1, k_b): 1} # C11
        add_term(u6, v6, w6)
        
        # P7
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(0, k_a): 1, (1, k_a): 1}
        w7 = {(0, k_a): 1, (1, k_b): -1} # C00, C11
        add_term(u7, v7, w7)

    # Schoolbook for a single column k
    def schoolbook_column(k):
        # C00 += A00 * B0k
        u1 = {(0,0): 1}
        v1 = {(0, k): 1}
        w1 = {(0, k): 1}
        add_term(u1, v1, w1)
        
        # C01 (which is C0k) += A01 * B1k
        # Wait, C_{i,k}. For single col k, output is C_{0,k} and C_{1,k}.
        # C_{0,k} = A_{0,0}B_{0,k} + A_{0,1}B_{1,k}
        # C_{1,k} = A_{1,0}B_{0,k} + A_{1,1}B_{1,k}
        
        # Term 2: A01 * B1k -> C0k
        u2 = {(0,1): 1}
        v2 = {(1, k): 1}
        w2 = {(0, k): 1}
        add_term(u2, v2, w2)
        
        # Term 3: A10 * B0k -> C1k
        u3 = {(1,0): 1}
        v3 = {(0, k): 1}
        w3 = {(1, k): 1}
        add_term(u3, v3, w3)
        
        # Term 4: A11 * B1k -> C1k
        u4 = {(1,1): 1}
        v4 = {(1, k): 1}
        w4 = {(1, k): 1}
        add_term(u4, v4, w4)

    # Apply to columns
    # Pairs: (0,1), (2,3), (4,5), (6,7), (8,9)
    for t in range(5):
        k_a = 2 * t
        k_b = 2 * t + 1
        strassen_block(k_a, k_b)
        
    # Single: 10
    schoolbook_column(10)
    
    return {"U": U, "V": V, "W": W}
