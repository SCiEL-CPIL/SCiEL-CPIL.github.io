import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=8
    
    # Total rank calculation:
    # 2x Strassen 2x2x8: 2 * 28 = 56
    # 2x Schoolbook 1x2x8: 2 * 16 = 32
    # Total = 88
    
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2 block
    # a_rows, a_cols: global indices for A block (2x2)
    # b_rows, b_cols: global indices for B block (2x2)
    # c_rows, c_cols: global indices for C block (2x2)
    def add_strassen(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        nonlocal col
        # A indices: (a_r0, a_c0), (a_r0, a_c0+1), (a_r0+1, a_c0), (a_r0+1, a_c0+1)
        # B indices: (b_r0, b_c0), (b_r0, b_c0+1), (b_r0+1, b_c0), (b_r0+1, b_c0+1)
        # C indices: (c_r0, c_c0), (c_r0, c_c0+1), (c_r0+1, c_c0), (c_r0+1, c_c0+1)
        
        # Map local (i,j) to global U index
        def u_idx(i, j): return a_r0 * m + (a_c0 + j)
        # Map local (i,j) to global V index
        def v_idx(i, j): return b_r0 * p + (b_c0 + j)
        # Map local (i,j) to global W index
        # W index is k*n + i. C is p x n.
        # c_r0, c_c0 are row, col in C (p x n).
        # So global W index is c_r0 * n + c_c0
        def w_idx(i, j): return c_r0 * n + (c_c0 + j)
        
        # Strassen terms
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        terms = [
            # (U coeffs, V coeffs, W coeffs)
            # U coeffs: dict {idx: val}
            # V coeffs: dict {idx: val}
            # W coeffs: dict {idx: val}
            
            # M1
            ({u_idx(0,0): 1, u_idx(1,1): 1}, {v_idx(0,0): 1, v_idx(1,1): 1}, {w_idx(0,0): 1, w_idx(1,1): 1}),
            # M4
            ({u_idx(1,1): 1}, {v_idx(1,0): 1, v_idx(0,0): -1}, {w_idx(0,0): 1}),
            # -M5
            ({u_idx(0,0): 1, u_idx(0,1): 1}, {v_idx(1,1): 1}, {w_idx(0,0): -1}),
            # M7
            ({u_idx(0,1): 1, u_idx(1,1): -1}, {v_idx(1,0): 1, v_idx(1,1): 1}, {w_idx(0,0): 1}),
            
            # M3
            ({u_idx(0,0): 1}, {v_idx(0,1): 1, v_idx(1,1): -1}, {w_idx(0,1): 1}),
            # M5
            ({u_idx(0,0): 1, u_idx(0,1): 1}, {v_idx(1,1): 1}, {w_idx(0,1): 1}),
            
            # M2
            ({u_idx(1,0): 1, u_idx(1,1): 1}, {v_idx(0,0): 1}, {w_idx(1,0): 1}),
            # M4
            ({u_idx(1,1): 1}, {v_idx(1,0): 1, v_idx(0,0): -1}, {w_idx(1,0): 1}),
            
            # M1
            ({u_idx(0,0): 1, u_idx(1,1): 1}, {v_idx(0,0): 1, v_idx(1,1): 1}, {w_idx(1,1): 1}),
            # -M2
            ({u_idx(1,0): 1, u_idx(1,1): 1}, {v_idx(0,0): 1}, {w_idx(1,1): -1}),
            # M3
            ({u_idx(0,0): 1}, {v_idx(0,1): 1, v_idx(1,1): -1}, {w_idx(1,1): 1}),
            # M6
            ({u_idx(1,0): 1, u_idx(0,0): -1}, {v_idx(0,0): 1, v_idx(0,1): 1}, {w_idx(1,1): 1}),
        ]
        
        # Wait, the standard Strassen decomposition has 7 products, but the reconstruction requires summing them.
        # The tensor decomposition corresponds to the 7 products.
        # Each product M_k contributes to multiple C elements.
        # So for each M_k, we have one column in U, V, W.
        # The W entry for that column will have coefficients for the C elements it affects.
        
        # Let's re-evaluate.
        # Product M1 = (A00+A11)(B00+B11).
        # U vector: A00: 1, A11: 1.
        # V vector: B00: 1, B11: 1.
        # W vector: C00: 1, C11: 1.
        
        # Product M2 = (A10+A11)B00.
        # U vector: A10: 1, A11: 1.
        # V vector: B00: 1.
        # W vector: C10: 1, C11: -1.
        
        # Product M3 = A00(B01-B11).
        # U vector: A00: 1.
        # V vector: B01: 1, B11: -1.
        # W vector: C01: 1, C11: 1.
        
        # Product M4 = A11(B10-B00).
        # U vector: A11: 1.
        # V vector: B10: 1, B00: -1.
        # W vector: C00: 1, C10: 1.
        
        # Product M5 = (A00+A01)B11.
        # U vector: A00: 1, A01: 1.
        # V vector: B11: 1.
        # W vector: C00: -1, C01: 1.
        
        # Product M6 = (A10-A00)(B00+B01).
        # U vector: A10: 1, A00: -1.
        # V vector: B00: 1, B01: 1.
        # W vector: C11: 1.
        
        # Product M7 = (A01-A11)(B10+B11).
        # U vector: A01: 1, A11: -1.
        # V vector: B10: 1, B11: 1.
        # W vector: C00: 1.
        
        # Let's verify C11: M1 - M2 + M3 + M6.
        # M1: C11=1. M2: C11=-1. M3: C11=1. M6: C11=1. Sum: 1-1+1+1=2?
        # Standard Strassen:
        # C11 = M1 - M2 + M3 + M6.
        # Let's check M6 definition.
        # M6 = (A10 - A00)(B00 + B01).
        # Does M6 contribute to C11?
        # C11 = A00 B01 + A01 B11 + A10 B01 + A11 B11 ? No.
        # C = A B.
        # C11 = A10 B01 + A11 B11.
        
        # Let's re-derive Strassen terms carefully.
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's check C11 with these formulas.
        # M1: A00B00 + A00B11 + A11B00 + A11B11
        # -M2: -A10B00 - A11B00
        # M3: A00B01 - A00B11
        # M6: A10B00 + A10B01 - A00B00 - A00B01
        # Sum:
        # A00B00: 1 - 1 = 0
        # A00B11: 1 - 1 = 0
        # A11B00: 1 - 1 = 0
        # A11B11: 1
        # A10B00: -1 + 1 = 0
        # A00B01: 1 - 1 = 0
        # A10B01: 1
        # Result: A11B11 + A10B01.
        # Correct for C11.
        
        # So the W coefficients are:
        # M1: C00: 1, C11: 1
        # M2: C10: 1, C11: -1
        # M3: C01: 1, C11: 1
        # M4: C00: 1, C10: 1
        # M5: C00: -1, C01: 1
        # M6: C11: 1
        # M7: C00: 1
        
        strassen_terms = [
            # (U dict, V dict, W dict)
            ({u_idx(0,0): 1, u_idx(1,1): 1}, {v_idx(0,0): 1, v_idx(1,1): 1}, {w_idx(0,0): 1, w_idx(1,1): 1}), # M1
            ({u_idx(1,0): 1, u_idx(1,1): 1}, {v_idx(0,0): 1}, {w_idx(1,0): 1, w_idx(1,1): -1}), # M2
            ({u_idx(0,0): 1}, {v_idx(0,1): 1, v_idx(1,1): -1}, {w_idx(0,1): 1, w_idx(1,1): 1}), # M3
            ({u_idx(1,1): 1}, {v_idx(1,0): 1, v_idx(0,0): -1}, {w_idx(0,0): 1, w_idx(1,0): 1}), # M4
            ({u_idx(0,0): 1, u_idx(0,1): 1}, {v_idx(1,1): 1}, {w_idx(0,0): -1, w_idx(0,1): 1}), # M5
            ({u_idx(1,0): 1, u_idx(0,0): -1}, {v_idx(0,0): 1, v_idx(0,1): 1}, {w_idx(1,1): 1}), # M6
            ({u_idx(0,1): 1, u_idx(1,1): -1}, {v_idx(1,0): 1, v_idx(1,1): 1}, {w_idx(0,0): 1}), # M7
        ]
        
        for u_dict, v_dict, w_dict in strassen_terms:
            for idx, val in u_dict.items():
                U[idx][col] = val
            for idx, val in v_dict.items():
                V[idx][col] = val
            for idx, val in w_dict.items():
                W[idx][col] = val
            col += 1

    # Helper to add Schoolbook 1x2x2 block (part of 1x2x8)
    # a_row, a_c0: global indices for A row segment (1x2)
    # b_r0, b_c0: global indices for B block (2x2)
    # c_row, c_c0: global indices for C row segment (1x2)
    def add_schoolbook_1x2(a_row, a_c0, b_r0, b_c0, c_row, c_c0):
        nonlocal col
        # A: (a_row, a_c0), (a_row, a_c0+1)
        # B: (b_r0, b_c0), (b_r0, b_c0+1), (b_r0+1, b_c0), (b_r0+1, b_c0+1)
        # C: (c_row, c_c0), (c_row, c_c0+1)
        
        # C0 = A0 B0 + A1 B2 (using row/col indices relative to block)
        # C1 = A0 B1 + A1 B3
        
        # Term 1: A0 B0 -> C0
        U[a_row * m + a_c0][col] = 1
        V[b_r0 * p + b_c0][col] = 1
        W[c_row * n + c_c0][col] = 1
        col += 1
        
        # Term 2: A1 B2 -> C0
        U[a_row * m + a_c0 + 1][col] = 1
        V[(b_r0 + 1) * p + b_c0][col] = 1
        W[c_row * n + c_c0][col] = 1
        col += 1
        
        # Term 3: A0 B1 -> C1
        U[a_row * m + a_c0][col] = 1
        V[b_r0 * p + b_c0 + 1][col] = 1
        W[c_row * n + c_c0 + 1][col] = 1
        col += 1
        
        # Term 4: A1 B3 -> C1
        U[a_row * m + a_c0 + 1][col] = 1
        V[(b_r0 + 1) * p + b_c0 + 1][col] = 1
        W[c_row * n + c_c0 + 1][col] = 1
        col += 1

    # Sub-problem 1: A[0:2, 0:2] B[0:2, 0:8] -> C[0:2, 0:8]
    # Split B into four 2x2 blocks: cols 0-1, 2-3, 4-5, 6-7
    for k in range(4):
        add_strassen(0, 0, 0, k*2, 0, k*2)
        
    # Sub-problem 2: A[0:2, 2:4] B[2:4, 0:8] -> C[0:2, 0:8]
    for k in range(4):
        add_strassen(0, 2, 2, k*2, 0, k*2)
        
    # Sub-problem 3: A[2, 0:2] B[0:2, 0:8] -> C[2, 0:8]
    for k in range(4):
        add_schoolbook_1x2(2, 0, 0, k*2, 2, k*2)
        
    # Sub-problem 4: A[2, 2:4] B[2:4, 0:8] -> C[2, 0:8]
    for k in range(4):
        add_schoolbook_1x2(2, 2, 2, k*2, 2, k*2)
        
    return {"U": U, "V": V, "W": W}
