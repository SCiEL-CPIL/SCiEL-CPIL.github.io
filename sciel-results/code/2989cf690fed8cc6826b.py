import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # List of (u_vec, v_vec, w_vec) for each rank-1 term
    # u_vec is length n*m = 8
    # v_vec is length m*p = 28
    # w_vec is length p*n = 14
    terms = []
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        # u_coeffs: dict {(i,j): val}
        # v_coeffs: dict {(j,k): val}
        # w_coeffs: dict {(k,i): val}
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        
        for (i, j), val in u_coeffs.items():
            u[i * m + j] = val
        for (j, k), val in v_coeffs.items():
            v[j * p + k] = val
        for (k, i), val in w_coeffs.items():
            w[k * n + i] = val
            
        terms.append((u, v, w))

    # Strassen for 2x2 by 2x2
    # Local indices 0,1 for rows/cols
    def strassen_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset):
        # a_row_offset: 0 or 1 (for A1 or A2)
        # a_col_offset: 0 or 1 (for left or right part of A) -- wait, A is 2x4.
        # A1 is cols 0,1. A2 is cols 2,3.
        # b_row_offset: 0 or 1 (for B1 or B2)
        # b_col_offset: offset in k for the 2x2 block in B
        
        # M1: (x00+x11)(y00+y11)
        # x00 is A[a_row_offset, a_col_offset+0] ? No.
        # Let's define local (r, c) for A block.
        # Global i = a_row_offset + r
        # Global j = a_col_offset + c
        # B block:
        # Global j = b_row_offset + r
        # Global k = b_col_offset + c
        
        # M1
        u1 = {(a_row_offset, a_col_offset): 1, (a_row_offset+1, a_col_offset+1): 1}
        v1 = {(b_row_offset, b_col_offset): 1, (b_row_offset+1, b_col_offset+1): 1}
        w1 = {(b_col_offset, a_row_offset): 1, (b_col_offset+1, a_row_offset+1): 1} # z00 and z11?
        # z00 = M1 + ...
        # z11 = M1 - ...
        # So W should have 1 at (k0, i0) and 1 at (k1, i1)?
        # z_{00} corresponds to k=b_col_offset, i=a_row_offset.
        # z_{11} corresponds to k=b_col_offset+1, i=a_row_offset+1.
        # Yes.
        add_term(u1, v1, w1)
        
        # M2: (x10+x11)y00
        u2 = {(a_row_offset+1, a_col_offset): 1, (a_row_offset+1, a_col_offset+1): 1}
        v2 = {(b_row_offset, b_col_offset): 1}
        w2 = {(b_col_offset, a_row_offset+1): 1, (b_col_offset+1, a_row_offset+1): -1} # z10 has M2, z11 has -M2
        add_term(u2, v2, w2)
        
        # M3: x00(y10-y11)
        u3 = {(a_row_offset, a_col_offset): 1}
        v3 = {(b_row_offset+1, b_col_offset): 1, (b_row_offset+1, b_col_offset+1): -1}
        w3 = {(b_col_offset+1, a_row_offset): 1, (b_col_offset+1, a_row_offset+1): 1} # z01 has M3, z11 has M3
        add_term(u3, v3, w3)
        
        # M4: x11(y01+y11)
        u4 = {(a_row_offset+1, a_col_offset+1): 1}
        v4 = {(b_row_offset, b_col_offset+1): 1, (b_row_offset+1, b_col_offset+1): 1}
        w4 = {(b_col_offset, a_row_offset): 1, (b_col_offset, a_row_offset+1): 1} # z00 has M4, z10 has M4
        add_term(u4, v4, w4)
        
        # M5: (x00+x01)y11
        u5 = {(a_row_offset, a_col_offset): 1, (a_row_offset, a_col_offset+1): 1}
        v5 = {(b_row_offset+1, b_col_offset+1): 1}
        w5 = {(b_col_offset+1, a_row_offset): -1, (b_col_offset+1, a_row_offset+1): 1} # z01 has M5, z00 has -M5
        add_term(u5, v5, w5)
        
        # M6: (x10-x00)(y00+y01)
        u6 = {(a_row_offset+1, a_col_offset): 1, (a_row_offset, a_col_offset): -1}
        v6 = {(b_row_offset, b_col_offset): 1, (b_row_offset, b_col_offset+1): 1}
        w6 = {(b_col_offset+1, a_row_offset+1): 1} # z11 has M6
        add_term(u6, v6, w6)
        
        # M7: (x01-x11)(y10+y11)
        u7 = {(a_row_offset, a_col_offset+1): 1, (a_row_offset+1, a_col_offset+1): -1}
        v7 = {(b_row_offset+1, b_col_offset): 1, (b_row_offset+1, b_col_offset+1): 1}
        w7 = {(b_col_offset, a_row_offset): 1} # z00 has M7
        add_term(u7, v7, w7)

    # Schoolbook for 2x2 by 2x1
    def schoolbook_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset):
        # z0 = x00*y0 + x01*y1
        # z1 = x10*y0 + x11*y1
        
        # Term 1: x00*y0 -> z0
        u1 = {(a_row_offset, a_col_offset): 1}
        v1 = {(b_row_offset, b_col_offset): 1}
        w1 = {(b_col_offset, a_row_offset): 1}
        add_term(u1, v1, w1)
        
        # Term 2: x01*y1 -> z0
        u2 = {(a_row_offset, a_col_offset+1): 1}
        v2 = {(b_row_offset+1, b_col_offset): 1}
        w2 = {(b_col_offset, a_row_offset): 1}
        add_term(u2, v2, w2)
        
        # Term 3: x10*y0 -> z1
        u3 = {(a_row_offset+1, a_col_offset): 1}
        v3 = {(b_row_offset, b_col_offset): 1}
        w3 = {(b_col_offset, a_row_offset+1): 1}
        add_term(u3, v3, w3)
        
        # Term 4: x11*y1 -> z1
        u4 = {(a_row_offset+1, a_col_offset+1): 1}
        v4 = {(b_row_offset+1, b_col_offset): 1}
        w4 = {(b_col_offset, a_row_offset+1): 1}
        add_term(u4, v4, w4)

    # Construct the full decomposition
    # A1 (rows 0-1, cols 0-1) * B1 (rows 0-1, cols 0-6)
    # B1 split into:
    # k=0,1: 2x2 block at col 0
    # k=2,3: 2x2 block at col 2
    # k=4,5: 2x2 block at col 4
    # k=6:   2x1 block at col 6
    
    # Block 1: A1 * B1[0:2, 0:2]
    strassen_block(0, 0, 0, 0)
    # Block 2: A1 * B1[0:2, 2:4]
    strassen_block(0, 0, 0, 2)
    # Block 3: A1 * B1[0:2, 4:6]
    strassen_block(0, 0, 0, 4)
    # Block 4: A1 * B1[0:2, 6:7]
    schoolbook_block(0, 0, 0, 6)
    
    # A2 (rows 0-1, cols 2-3) * B2 (rows 2-3, cols 0-6)
    # B2 split into:
    # k=0,1: 2x2 block at col 0
    # k=2,3: 2x2 block at col 2
    # k=4,5: 2x2 block at col 4
    # k=6:   2x1 block at col 6
    
    # Block 5: A2 * B2[2:4, 0:2]
    strassen_block(0, 2, 2, 0)
    # Block 6: A2 * B2[2:4, 2:4]
    strassen_block(0, 2, 2, 2)
    # Block 7: A2 * B2[2:4, 4:6]
    strassen_block(0, 2, 2, 4)
    # Block 8: A2 * B2[2:4, 6:7]
    schoolbook_block(0, 2, 2, 6)
    
    # Assemble matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t, (u, v, w) in enumerate(terms):
        for i in range(n * m):
            U[i][t] = u[i]
        for j in range(m * p):
            V[j][t] = v[j]
        for k in range(p * n):
            W[k][t] = w[k]
            
    return {"U": U, "V": V, "W": W}
