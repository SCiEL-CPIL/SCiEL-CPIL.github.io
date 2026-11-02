import numpy as np
import time

def construct():
    n, m, p = 4, 4, 6
    
    # Target dimensions
    U_dim = n * m  # 16
    V_dim = m * p  # 24
    W_dim = p * n  # 24
    
    # We will use Rank 77
    # Rank 49 for 4x4x4
    # Rank 28 for 4x4x2
    total_rank = 77
    
    U = [[0] * total_rank for _ in range(U_dim)]
    V = [[0] * total_rank for _ in range(V_dim)]
    W = [[0] * total_rank for _ in range(W_dim)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # U, V, W for one 2x2x2 block
    # Indices: A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -- Note: W index is k*n+i.
    # For 2x2x2, n=2, p=2.
    # C index k*n+i.
    # C00 (k=0, i=0) -> 0
    # C01 (k=1, i=0) -> 2
    # C10 (k=0, i=1) -> 1
    # C11 (k=1, i=1) -> 3
    
    # Let's define the 7 terms for 2x2x2
    # Each term is (u_vec, v_vec, w_vec)
    # u_vec, v_vec, w_vec are length 4.
    
    terms_2x2x2 = [
        # M1
        ([1, 0, 1, 1], [1, 0, 1, 1], [1, 0, 0, -1]), # Wait, C11 = M1...
        # Let's be precise.
        # M1 = (a+d)(e+h)
        # U: a=1, d=1. V: e=1, h=1.
        # W: C11 += 1, C22 += -1? No.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So for M1:
        # U: [1, 0, 0, 1] (a, d)
        # V: [1, 0, 0, 1] (e, h)
        # W: C11 gets +1, C22 gets +1?
        # C22 = M1... so C22 gets +1.
        # So W: [1, 0, 0, 1] (C11, C22)
        
        # M2 = (c+d)e
        # U: [0, 0, 1, 1] (c, d)
        # V: [1, 0, 0, 0] (e)
        # W: C21 gets +1. C22 gets -1.
        # W: [0, 1, 0, -1] (C21, C22)
        
        # M3 = a(f-h)
        # U: [1, 0, 0, 0] (a)
        # V: [0, 1, 0, -1] (f, h)
        # W: C12 gets +1. C22 gets +1.
        # W: [0, 1, 0, 1] (C12, C22)
        
        # M4 = d(g-e)
        # U: [0, 0, 0, 1] (d)
        # V: [0, 0, 1, -1] (g, e) -- Wait, g is B[1,0], e is B[0,0].
        # V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
        # e=0, f=1, g=2, h=3.
        # M4 = d(g-e). U: d (idx 3). V: g (idx 2), -e (idx 0).
        # V: [-1, 0, 1, 0].
        # W: C11 gets +1. C21 gets +1.
        # W: [1, 0, 1, 0] (C11, C21)
        
        # M5 = (a+b)h
        # U: [1, 1, 0, 0] (a, b)
        # V: [0, 0, 0, 1] (h)
        # W: C11 gets -1. C12 gets +1.
        # W: [-1, 1, 0, 0] (C11, C12)
        
        # M6 = (c-a)(e+f)
        # U: [1, 0, -1, 0] (a, c) -- c-a. c is idx 2, a is idx 0.
        # U: [-1, 0, 1, 0].
        # V: [1, 1, 0, 0] (e, f)
        # W: C22 gets +1.
        # W: [0, 0, 0, 1] (C22)
        
        # M7 = (b-d)(g+h)
        # U: [0, 1, 0, -1] (b, d)
        # V: [0, 0, 1, 1] (g, h)
        # W: C11 gets +1.
        # W: [1, 0, 0, 0] (C11)
    ]
    
    # Corrected terms_2x2x2
    terms_2x2x2 = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    def add_block(U, V, W, u_offsets, v_offsets, w_offsets, terms, start_col):
        """
        Adds a block decomposition to U, V, W.
        u_offsets: list of 4 global indices for A block.
        v_offsets: list of 4 global indices for B block.
        w_offsets: list of 4 global indices for C block.
        terms: list of (u_vec, v_vec, w_vec) for 2x2x2.
        start_col: column index in U,V,W to start writing.
        Returns new start_col.
        """
        for t in terms:
            u_vec, v_vec, w_vec = t
            for k in range(4):
                if u_vec[k] != 0:
                    U[u_offsets[k]][start_col] += u_vec[k]
                if v_vec[k] != 0:
                    V[v_offsets[k]][start_col] += v_vec[k]
                if w_vec[k] != 0:
                    W[w_offsets[k]][start_col] += w_vec[k]
            start_col += 1
        return start_col

    # Part 1: 4x4x4
    # Split A (4x4) into 4 blocks of 2x2.
    # A11 (0,0), A12 (0,1), A21 (1,0), A22 (1,1)
    # Global A indices:
    # A11: 0,1,4,5
    # A12: 2,3,6,7
    # A21: 8,9,12,13
    # A22: 10,11,14,15
    
    # Split B (4x4) into 4 blocks of 2x2.
    # B11 (0,0), B12 (0,1), B21 (1,0), B22 (1,1)
    # Global B indices (m=4, p=4):
    # B11: 0,1,4,5
    # B12: 2,3,6,7
    # B21: 8,9,12,13
    # B22: 10,11,14,15
    
    # Split C (4x4) into 4 blocks of 2x2.
    # C11 (0,0), C12 (0,1), C21 (1,0), C22 (1,1)
    # Global C indices (p=4, n=4): W[k*n+i]
    # C11 (k=0,1; i=0,1):
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 4
    # k=1, i=1 -> 5
    # So C11: 0,1,4,5
    # C12 (k=2,3; i=0,1):
    # k=2, i=0 -> 8
    # k=2, i=1 -> 9
    # k=3, i=0 -> 12
    # k=3, i=1 -> 13
    # So C12: 8,9,12,13
    # C21 (k=0,1; i=2,3):
    # k=0, i=2 -> 2
    # k=0, i=3 -> 3
    # k=1, i=2 -> 6
    # k=1, i=3 -> 7
    # So C21: 2,3,6,7
    # C22 (k=2,3; i=2,3):
    # k=2, i=2 -> 10
    # k=2, i=3 -> 11
    # k=3, i=2 -> 14
    # k=3, i=3 -> 15
    # So C22: 10,11,14,15
    
    col = 0
    
    # Strassen for 4x4x4
    # M1 = (A11+A22)(B11+B22)
    # This requires summing A11 and A22.
    # In the tensor decomposition, we can't just sum matrices in U.
    # We need to express (A11+A22) as a linear combination of basis vectors.
    # Actually, the decomposition is linear.
    # M1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
    # This is 4 schoolbook multiplications.
    # But Strassen combines them into 1 multiplication of sums.
    # The tensor decomposition must reflect this.
    # U for M1 should have non-zeros in A11 and A22 positions.
    # V for M1 should have non-zeros in B11 and B22 positions.
    # W for M1 should have non-zeros in C11 and C22 positions?
    # No, M1 contributes to C11 and C22?
    # In 2x2x2, M1 = (a+d)(e+h) contributed to C11 and C22.
    # So for 4x4x4, M1 will contribute to C11 and C22.
    
    # Let's construct the 7 terms for 4x4x4 directly from Strassen logic.
    
    # Term 1: M1 = (A11+A22)(B11+B22)
    # U: A11 (1,1,1,1) at 0,1,4,5. A22 (1,1,1,1) at 10,11,14,15.
    # V: B11 (1,1,1,1) at 0,1,4,5. B22 (1,1,1,1) at 10,11,14,15.
    # W: C11 (1,1,1,1) at 0,1,4,5. C22 (1,1,1,1) at 10,11,14,15.
    # Wait, in 2x2x2, M1 was [1,0,0,1] for U (a,d).
    # Here A11 is 2x2. So all 4 entries are 1.
    
    # Let's write a helper to set a 2x2 block in U/V/W.
    
    def set_block_2x2(matrix, offsets, val, col):
        matrix[offsets[0]][col] += val
        matrix[offsets[1]][col] += val
        matrix[offsets[2]][col] += val
        matrix[offsets[3]][col] += val

    # Offsets
    A11 = [0, 1, 4, 5]
    A12 = [2, 3, 6, 7]
    A21 = [8, 9, 12, 13]
    A22 = [10, 11, 14, 15]
    
    B11 = [0, 1, 4, 5]
    B12 = [2, 3, 6, 7]
    B21 = [8, 9, 12, 13]
    B22 = [10, 11, 14, 15]
    
    C11 = [0, 1, 4, 5]
    C12 = [8, 9, 12, 13]
    C21 = [2, 3, 6, 7]
    C22 = [10, 11, 14, 15]
    
    # M1
    set_block_2x2(U, A11, 1, col)
    set_block_2x2(U, A22, 1, col)
    set_block_2x2(V, B11, 1, col)
    set_block_2x2(V, B22, 1, col)
    set_block_2x2(W, C11, 1, col)
    set_block_2x2(W, C22, 1, col)
    col += 1
    
    # M2 = (A21+A22)B11
    set_block_2x2(U, A21, 1, col)
    set_block_2x2(U, A22, 1, col)
    set_block_2x2(V, B11, 1, col)
    set_block_2x2(W, C21, 1, col)
    set_block_2x2(W, C22, -1, col)
    col += 1
    
    # M3 = A11(B12-B22)
    set_block_2x2(U, A11, 1, col)
    set_block_2x2(V, B12, 1, col)
    set_block_2x2(V, B22, -1, col)
    set_block_2x2(W, C12, 1, col)
    set_block_2x2(W, C22, 1, col)
    col += 1
    
    # M4 = A22(B21-B11)
    set_block_2x2(U, A22, 1, col)
    set_block_2x2(V, B21, 1, col)
    set_block_2x2(V, B11, -1, col)
    set_block_2x2(W, C11, 1, col)
    set_block_2x2(W, C21, 1, col)
    col += 1
    
    # M5 = (A11+A12)B22
    set_block_2x2(U, A11, 1, col)
    set_block_2x2(U, A12, 1, col)
    set_block_2x2(V, B22, 1, col)
    set_block_2x2(W, C11, -1, col)
    set_block_2x2(W, C12, 1, col)
    col += 1
    
    # M6 = (A21-A11)(B11+B12)
    set_block_2x2(U, A21, 1, col)
    set_block_2x2(U, A11, -1, col)
    set_block_2x2(V, B11, 1, col)
    set_block_2x2(V, B12, 1, col)
    set_block_2x2(W, C22, 1, col)
    col += 1
    
    # M7 = (A12-A22)(B21+B22)
    set_block_2x2(U, A12, 1, col)
    set_block_2x2(U, A22, -1, col)
    set_block_2x2(V, B21, 1, col)
    set_block_2x2(V, B22, 1, col)
    set_block_2x2(W, C11, 1, col)
    col += 1
    
    # Part 2: 4x4x2
    # B is 4x2. Split into B1 (4x1) and B2 (4x1)?
    # Or B1 (2x2) and B2 (2x2) rows?
    # B (4x2) -> B_top (2x2), B_bot (2x2).
    # A (4x4) -> A11, A12, A21, A22 (2x2).
    # C (4x2) -> C_top (2x2), C_bot (2x2).
    # C_top = A11 B_top + A12 B_bot
    # C_bot = A21 B_top + A22 B_bot
    
    # B_top indices (m=4, p=2):
    # Rows 0,1. Cols 0,1.
    # 0,1, 4,5
    # B_bot indices:
    # Rows 2,3. Cols 0,1.
    # 8,9, 12,13
    
    # C_top indices (p=2, n=4): W[k*n+i]
    # k=0,1. i=0,1.
    # 0,1, 4,5
    # C_bot indices:
    # k=0,1. i=2,3.
    # 2,3, 6,7
    
    # Term 1: A11 B_top
    set_block_2x2(U, A11, 1, col)
    set_block_2x2(V, B_top, 1, col)
    set_block_2x2(W, C_top, 1, col)
    col += 1
    
    # Term 2: A12 B_bot
    set_block_2x2(U, A12, 1, col)
    set_block_2x2(V, B_bot, 1, col)
    set_block_2x2(W, C_top, 1, col)
    col += 1
    
    # Term 3: A21 B_top
    set_block_2x2(U, A21, 1, col)
    set_block_2x2(V, B_top, 1, col)
    set_block_2x2(W, C_bot, 1, col)
    col += 1
    
    # Term 4: A22 B_bot
    set_block_2x2(U, A22, 1, col)
    set_block_2x2(V, B_bot, 1, col)
    set_block_2x2(W, C_bot, 1, col)
    col += 1
    
    return {"U": U, "V": V, "W": W}
