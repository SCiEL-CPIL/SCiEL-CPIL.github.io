import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 114 construction
    # Split m=5 into m1=2, m2=3
    # Part 1: A1 (4x2) * B1 (2x6) -> Rank 42
    # Part 2: A2 (4x3) * B2 (3x6) -> Rank 72
    
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_entries, v_entries, w_entries):
        nonlocal col
        for idx, val in u_entries:
            U[idx][col] += val
        for idx, val in v_entries:
            V[idx][col] += val
        for idx, val in w_entries:
            W[idx][col] += val
        col += 1

    # Part 1: 4x2 * 2x6
    # A1 indices: i in 0..3, j in 0..1 (global j offset 0)
    # B1 indices: j in 0..1, k in 0..5 (global j offset 0, k offset 0)
    # C indices: i in 0..3, k in 0..5
    
    # Split A1 into At (rows 0,1) and Ab (rows 2,3)
    # Split B1 into B1_1 (cols 0,1), B1_2 (cols 2,3), B1_3 (cols 4,5)
    
    # Strassen coefficients for 2x2 multiplication
    # Inputs: A (2x2), B (2x2)
    # A indices: (0,0), (0,1), (1,0), (1,1) -> local (0,0)..(1,1)
    # B indices: (0,0)..(1,1)
    # Output C: (0,0)..(1,1)
    
    def strassen_terms(a_row_offset, b_col_offset, c_row_offset, c_col_offset):
        # a_row_offset: global row index of A[0,0]
        # b_col_offset: global col index of B[0,0]
        # c_row_offset: global row index of C[0,0]
        # c_col_offset: global col index of C[0,0]
        
        # Local indices
        # A: (0,0) at (a_row_offset, 0), (0,1) at (a_row_offset, 1)
        #     (1,0) at (a_row_offset+1, 0), (1,1) at (a_row_offset+1, 1)
        # B: (0,0) at (0, b_col_offset), (0,1) at (0, b_col_offset+1)
        #     (1,0) at (1, b_col_offset), (1,1) at (1, b_col_offset+1)
        # C: (0,0) at (c_row_offset, c_col_offset) ...
        
        # Note: In our global indexing:
        # U index = i * m + j
        # V index = j * p + k
        # W index = k * n + i
        
        # For Part 1 (m1=2):
        # A1 is 4x2. Global j is 0,1.
        # At: rows 0,1. Global i=0,1. j=0,1.
        # Ab: rows 2,3. Global i=2,3. j=0,1.
        # B1 is 2x6. Global j=0,1. k=0..5.
        
        # We are computing At * B1_block
        # At is 2x2. B1_block is 2x2.
        
        # Term 1: (A00+A11)(B00+B11)
        # U: (a_row_offset, 0) coeff 1, (a_row_offset+1, 1) coeff 1
        # Wait, A indices in U are i*m+j.
        # For At (i=0,1), j=0,1.
        # A00 is i=0, j=0 -> U[0*2+0] = U[0]
        # A11 is i=1, j=1 -> U[1*2+1] = U[3]
        # A10 is i=1, j=0 -> U[1*2+0] = U[2]
        # A01 is i=0, j=1 -> U[0*2+1] = U[1]
        
        # For B1_block (j=0,1), k=offset, offset+1.
        # B00 is j=0, k=offset -> V[0*6+offset]
        # B11 is j=1, k=offset+1 -> V[1*6+offset+1]
        # B01 is j=0, k=offset+1 -> V[0*6+offset+1]
        # B10 is j=1, k=offset -> V[1*6+offset]
        
        # C indices: i=c_row_offset, c_row_offset+1. k=c_col_offset, c_col_offset+1.
        # W index = k*n + i.
        # C00: k=c_col_offset, i=c_row_offset -> W[c_col_offset*4 + c_row_offset]
        # C01: k=c_col_offset+1, i=c_row_offset -> W[(c_col_offset+1)*4 + c_row_offset]
        # C10: k=c_col_offset, i=c_row_offset+1 -> W[c_col_offset*4 + c_row_offset+1]
        # C11: k=c_col_offset+1, i=c_row_offset+1 -> W[(c_col_offset+1)*4 + c_row_offset+1]
        
        # We will generate the 7 terms for Strassen
        
        # 1. (A00+A11)(B00+B11)
        u1 = [(a_row_offset*m + 0, 1), (a_row_offset+m + 1, 1)] # i=0,j=0 and i=1,j=1
        # Wait, a_row_offset is global i.
        # If At, a_row_offset=0. i=0,1.
        # U indices: 0*2+0=0, 1*2+1=3.
        v1 = [(0*p + b_col_offset, 1), (1*p + b_col_offset+1, 1)]
        w1 = [(c_col_offset*n + c_row_offset, 1), (c_col_offset*n + c_row_offset+1, 1), 
              ((c_col_offset+1)*n + c_row_offset, 1), ((c_col_offset+1)*n + c_row_offset+1, 1)]
        # Signs for C:
        # C00: +, C01: 0, C10: 0, C11: -
        # Actually, Strassen formulas:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1 = (A00+A11)(B00+B11)
        # W coeffs: C00: 1, C01: 0, C10: 0, C11: -1? No.
        # C00 gets +M1. C11 gets +M1? No.
        # C00 = M1 + ...
        # C11 = M1 - ...
        # So W for M1: C00: 1, C11: 1?
        # Let's check standard Strassen.
        # P1 = (A11 + A22)(B11 + B22)
        # P2 = (A21 + A22)B11
        # P3 = A11(B21 - B22)
        # P4 = A22(B21 - B11)
        # P5 = (A11 + A12)B22
        # P6 = (A12 - A11)(B11 + B12)
        # P7 = (A21 - A12)(B12 + B22)
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        
        # So for P1 (M1):
        # C11 (C00 in 0-indexed) coeff 1.
        # C22 (C11 in 0-indexed) coeff 1.
        # C12, C21 coeff 0.
        
        add_term(u1, v1, [
            (c_col_offset*n + c_row_offset, 1),
            ((c_col_offset+1)*n + c_row_offset+1, 1)
        ])
        
        # 2. (A10+A11)B00
        u2 = [(a_row_offset+m + 0, 1), (a_row_offset+m + 1, 1)] # i=1, j=0 and i=1, j=1
        v2 = [(0*p + b_col_offset, 1)]
        # C21 (C10 in 0-indexed) coeff 1.
        add_term(u2, v2, [
            (c_col_offset*n + c_row_offset+1, 1)
        ])
        
        # 3. A00(B01-B11)
        u3 = [(a_row_offset*m + 0, 1)]
        v3 = [(0*p + b_col_offset+1, 1), (1*p + b_col_offset+1, -1)]
        # C12 (C01 in 0-indexed) coeff 1.
        # C22 (C11 in 0-indexed) coeff 1.
        add_term(u3, v3, [
            ((c_col_offset+1)*n + c_row_offset, 1),
            ((c_col_offset+1)*n + c_row_offset+1, 1)
        ])
        
        # 4. A11(B10-B00)
        u4 = [(a_row_offset+m + 1, 1)]
        v4 = [(1*p + b_col_offset, 1), (0*p + b_col_offset, -1)]
        # C11 (C00 in 0-indexed) coeff 1.
        # C21 (C10 in 0-indexed) coeff 1.
        add_term(u4, v4, [
            (c_col_offset*n + c_row_offset, 1),
            (c_col_offset*n + c_row_offset+1, 1)
        ])
        
        # 5. (A00+A01)B11
        u5 = [(a_row_offset*m + 0, 1), (a_row_offset*m + 1, 1)]
        v5 = [(1*p + b_col_offset+1, 1)]
        # C11 (C00 in 0-indexed) coeff -1.
        # C12 (C01 in 0-indexed) coeff 1.
        add_term(u5, v5, [
            (c_col_offset*n + c_row_offset, -1),
            ((c_col_offset+1)*n + c_row_offset, 1)
        ])
        
        # 6. (A01-A00)(B00+B01)
        u6 = [(a_row_offset*m + 1, 1), (a_row_offset*m + 0, -1)]
        v6 = [(0*p + b_col_offset, 1), (0*p + b_col_offset+1, 1)]
        # C22 (C11 in 0-indexed) coeff 1.
        add_term(u6, v6, [
            ((c_col_offset+1)*n + c_row_offset+1, 1)
        ])
        
        # 7. (A10-A01)(B01+B11)
        u7 = [(a_row_offset+m + 0, 1), (a_row_offset*m + 1, -1)]
        v7 = [(0*p + b_col_offset+1, 1), (1*p + b_col_offset+1, 1)]
        # C11 (C00 in 0-indexed) coeff 1.
        add_term(u7, v7, [
            (c_col_offset*n + c_row_offset, 1)
        ])

    # Apply Strassen to At * B1_1, B1_2, B1_3
    # At: a_row_offset = 0
    # B1_1: b_col_offset = 0
    # C: c_row_offset = 0, c_col_offset = 0
    strassen_terms(0, 0, 0, 0)
    strassen_terms(0, 2, 0, 2)
    strassen_terms(0, 4, 0, 4)
    
    # Apply Strassen to Ab * B1_1, B1_2, B1_3
    # Ab: a_row_offset = 2
    strassen_terms(2, 0, 2, 0)
    strassen_terms(2, 2, 2, 2)
    strassen_terms(2, 4, 2, 4)
    
    # Part 2: 4x3 * 3x6
    # A2: i in 0..3, j in 2..4 (global j offset 2)
    # B2: j in 2..4, k in 0..5
    # C: i in 0..3, k in 0..5
    
    # Split A2 into At2 (i=0,1) and Ab2 (i=2,3)
    # For each, we do schoolbook on 2x3 * 3x6
    # 2x3 * 3x6 can be split into 3 blocks of 2x3 * 3x2?
    # Or just schoolbook 2x3 * 3x6 = 36 terms.
    
    def schoolbook_block(a_row_offset, j_start, b_col_offset, c_row_offset, c_col_offset, m_size, p_size):
        # a_row_offset: global i start
        # j_start: global j start
        # b_col_offset: global k start? No, B is j x k.
        # B2 is 3x6. j is 2,3,4. k is 0..5.
        # We iterate over all i, j, k in this block.
        
        # i from a_row_offset to a_row_offset+1
        # j from j_start to j_start+m_size-1
        # k from b_col_offset to b_col_offset+p_size-1
        
        for i in range(a_row_offset, a_row_offset + 2):
            for j in range(j_start, j_start + m_size):
                for k in range(b_col_offset, b_col_offset + p_size):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])

    # At2: i=0,1. j=2,3,4.
    # B2: j=2,3,4. k=0..5.
    # C: i=0,1. k=0..5.
    schoolbook_block(0, 2, 0, 0, 0, 3, 6)
    
    # Ab2: i=2,3. j=2,3,4.
    # C: i=2,3. k=0..5.
    schoolbook_block(2, 2, 0, 2, 0, 3, 6)

    return {"U": U, "V": V, "W": W}
