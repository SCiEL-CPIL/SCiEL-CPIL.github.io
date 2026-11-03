import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def _strassen_2x2x2():
    # Returns U, V, W for 2x2x2 multiplication with rank 7
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Indices:
    # A: 0=a, 1=b, 2=c, 3=d
    # B: 0=e, 1=f, 2=g, 3=h
    # C: 0=C11, 1=C12, 2=C21, 3=C22
    
    U = []
    V = []
    W = []
    
    # M1 = (a+d)(e+h)
    U.append([1, 0, 0, 1])
    V.append([1, 0, 0, 1])
    W.append([1, 0, 0, 0]) # C11
    # W for M1 also goes to C22
    # We need to add contributions to W for each C component.
    # It's easier to build the full U, V, W matrices directly.
    
    # Let's just list the 7 terms for one block.
    # Each term is (u, v, w)
    # u, v, w are vectors of length 4.
    
    terms = []
    # M1
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1])) # C11, C22? No.
    # C11 gets M1. C22 gets M1.
    # So W vector for M1 is [1, 0, 0, 1] (C11 and C22).
    
    # M2 = (c+d)e
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0])) # C21
    
    # M3 = a(f-h)
    terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0])) # C12
    
    # M4 = d(g-e)
    terms.append(([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0])) # C11, C21
    
    # M5 = (a+b)h
    terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0])) # C11, C12
    
    # M6 = (c-a)(e+f)
    terms.append([[-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]]) # C22
    
    # M7 = (b-d)(g+h)
    terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1])) # C11, C22
    
    return terms

def construct():
    n, m, p = DIMENSIONS
    # We will build U, V, W as lists of lists.
    # U size: n*m = 12
    # V size: m*p = 24
    # W size: p*n = 18
    
    U = []
    V = []
    W = []
    
    # Helper to add a block
    # block_u: list of 4 floats
    # block_v: list of 4 floats
    # block_w: list of 4 floats
    # offset_u, offset_v, offset_w: starting indices in the global vectors
    
    def add_term(u_vec, v_vec, w_vec, off_u, off_v, off_w):
        # u_vec is length 4, corresponds to A indices
        # v_vec is length 4, corresponds to B indices (2x2 block)
        # w_vec is length 4, corresponds to C indices (2x2 block)
        
        # Create full length vectors
        full_u = [0] * (n * m)
        full_v = [0] * (m * p)
        full_w = [0] * (p * n)
        
        for i in range(4):
            if u_vec[i] != 0:
                full_u[off_u + i] = u_vec[i]
            if v_vec[i] != 0:
                full_v[off_v + i] = v_vec[i]
            if w_vec[i] != 0:
                full_w[off_w + i] = w_vec[i]
                
        U.append(full_u)
        V.append(full_v)
        W.append(full_w)
        
    strassen_terms = _strassen_2x2x2()
    
    # Part 1: C1 = A1 B (2x4 * 4x6)
    # A1 is rows 0,1 of A.
    # Split B into 3 blocks of 2 columns? No, B is 4x6.
    # We split M (columns of A, rows of B) into 2+2.
    # A1 = [A11, A12] (2x2, 2x2)
    # B = [B1, B2] (4x6, 4x6) -- NO, B is not split this way.
    # B is 4x6. A1 is 2x4.
    # C1 = A11 * B_left + A12 * B_right ?
    # B_left is 4x6, B_right is 4x6? No.
    # A11 is 2x2 (cols 0,1 of A). A12 is 2x2 (cols 2,3 of A).
    # B is 4x6.
    # A11 * B uses rows 0,1 of B.
    # A12 * B uses rows 2,3 of B.
    # So C1 = A11 * B[0:2, :] + A12 * B[2:4, :].
    # B[0:2, :] is 2x6. B[2:4, :] is 2x6.
    # We need to compute 2x2 * 2x6.
    # This is 3 * (2x2 * 2x2).
    
    # Block 1: A11 (2x2) * B1 (2x6)
    # A11 is at A indices 0,1.
    # B1 is at B indices 0..5 (rows 0,1 of B).
    # C1 part 1 is at C indices 0..5 (rows 0,1 of C).
    
    # A11 indices in A: 0, 1, 2, 3 (row-major: 00, 01, 10, 11)
    # Wait, A is 3x4.
    # A1 is 2x4 (rows 0,1).
    # A11 is 2x2 (rows 0,1, cols 0,1).
    # Indices in A (row-major):
    # 0: (0,0), 1: (0,1)
    # 2: (1,0), 3: (1,1)
    
    # B is 4x6.
    # B1 is 2x6 (rows 0,1).
    # Indices in B (row-major):
    # 0: (0,0), 1: (0,1), 2: (0,2), 3: (0,3), 4: (0,4), 5: (0,5)
    # 6: (1,0), 7: (1,1), 8: (1,2), 9: (1,3), 10: (1,4), 11: (1,5)
    
    # C1 is 2x6 (rows 0,1).
    # Indices in C (row-major, but C is p x n? No, C is n x p).
    # Task says W[k*n+i, r]. C is n x p.
    # C indices: i (row), k (col).
    # Index = k*n + i.
    # C1 rows 0,1. Cols 0..5.
    # Indices:
    # k=0: i=0->0, i=1->1
    # k=1: i=0->3, i=1->4
    # k=2: i=0->6, i=1->7
    # k=3: i=0->9, i=1->10
    # k=4: i=0->12, i=1->13
    # k=5: i=0->15, i=1->16
    
    # We need to map the 2x2 Strassen block to these.
    # Strassen block computes 2x2 * 2x2.
    # We have 2x2 * 2x6.
    # We split the 6 columns into 3 groups of 2.
    
    # Group 0: Cols 0,1.
    # B cols 0,1: indices 0,1 (row 0), 6,7 (row 1).
    # B_vec for this block: [B(0,0), B(0,1), B(1,0), B(1,1)] -> indices 0, 1, 6, 7.
    # C cols 0,1: indices 0, 1 (k=0), 3, 4 (k=1).
    # C_vec for this block: [C(0,0), C(0,1), C(1,0), C(1,1)] -> indices 0, 3, 1, 4.
    
    # We need a helper to add a Strassen block given the index mappings.
    
    def add_strassen_block(a_indices, b_indices, c_indices):
        # a_indices: 4 indices for A
        # b_indices: 4 indices for B
        # c_indices: 4 indices for C
        for u, v, w in strassen_terms:
            full_u = [0] * (n * m)
            full_v = [0] * (m * p)
            full_w = [0] * (p * n)
            
            for j in range(4):
                if u[j] != 0:
                    full_u[a_indices[j]] = u[j]
                if v[j] != 0:
                    full_v[b_indices[j]] = v[j]
                if w[j] != 0:
                    full_w[c_indices[j]] = w[j]
                    
            U.append(full_u)
            V.append(full_v)
            W.append(full_w)
            
    # A11 indices: 0, 1, 2, 3
    # B1 (rows 0,1) indices for cols 0,1: 0, 1, 6, 7
    # C1 (rows 0,1) indices for cols 0,1: 0, 3, 1, 4
    
    # Block 1.1: Cols 0,1
    add_strassen_block([0, 1, 2, 3], [0, 1, 6, 7], [0, 3, 1, 4])
    
    # Block 1.2: Cols 2,3
    # B cols 2,3: indices 2, 3 (row 0), 8, 9 (row 1)
    # C cols 2,3: indices 6, 9 (k=2), 7, 10 (k=3) -> 6, 7, 9, 10?
    # C index = k*n + i.
    # k=2: i=0->6, i=1->7.
    # k=3: i=0->9, i=1->10.
    # So c_indices: [6, 9, 7, 10]
    add_strassen_block([0, 1, 2, 3], [2, 3, 8, 9], [6, 9, 7, 10])
    
    # Block 1.3: Cols 4,5
    # B cols 4,5: indices 4, 5 (row 0), 10, 11 (row 1)
    # C cols 4,5: indices 12, 15 (k=4), 13, 16 (k=5)
    # c_indices: [12, 15, 13, 16]
    add_strassen_block([0, 1, 2, 3], [4, 5, 10, 11], [12, 15, 13, 16])
    
    # Now A12 (rows 0,1, cols 2,3 of A)
    # A12 indices: 2, 3 (row 0), 6, 7 (row 1)
    # Wait, A is 3x4.
    # Row 0: 0, 1, 2, 3
    # Row 1: 4, 5, 6, 7
    # Row 2: 8, 9, 10, 11
    # A11 (cols 0,1): 0, 1, 4, 5.
    # A12 (cols 2,3): 2, 3, 6, 7.
    
    # My previous A11 indices were [0, 1, 2, 3] which is WRONG.
    # A is row-major.
    # A11 (rows 0,1, cols 0,1):
    # (0,0)=0, (0,1)=1
    # (1,0)=4, (1,1)=5
    # So a_indices for A11: [0, 1, 4, 5]
    
    # A12 (rows 0,1, cols 2,3):
    # (0,2)=2, (0,3)=3
    # (1,2)=6, (1,3)=7
    # So a_indices for A12: [2, 3, 6, 7]
    
    # B is 4x6.
    # B1 (rows 0,1) used for A11.
    # B2 (rows 2,3) used for A12.
    # B2 indices:
    # Row 2: 12, 13, 14, 15, 16, 17
    # Row 3: 18, 19, 20, 21, 22, 23
    
    # C1 (rows 0,1) is the same for both A11 and A12 contributions?
    # Yes, C1 = A11 B1 + A12 B2.
    # So we add more terms to the SAME C indices.
    
    # Block 1.4: A12 * B2 (cols 0,1)
    # b_indices for B2 cols 0,1: 12, 13 (row 2), 18, 19 (row 3)
    # c_indices for C1 cols 0,1: 0, 3, 1, 4
    add_strassen_block([2, 3, 6, 7], [12, 13, 18, 19], [0, 3, 1, 4])
    
    # Block 1.5: A12 * B2 (cols 2,3)
    # b_indices: 14, 15 (row 2), 20, 21 (row 3)
    # c_indices: 6, 9, 7, 10
    add_strassen_block([2, 3, 6, 7], [14, 15, 20, 21], [6, 9, 7, 10])
    
    # Block 1.6: A12 * B2 (cols 4,5)
    # b_indices: 16, 17 (row 2), 22, 23 (row 3)
    # c_indices: 12, 15, 13, 16
    add_strassen_block([2, 3, 6, 7], [16, 17, 22, 23], [12, 15, 13, 16])
    
    # Part 2: C2 = A2 B (1x4 * 4x6)
    # A2 is row 2 of A.
    # Indices: 8, 9, 10, 11.
    # B is 4x6.
    # C2 is row 2 of C.
    # Indices:
    # k=0: i=2 -> 2
    # k=1: i=2 -> 5
    # k=2: i=2 -> 8
    # k=3: i=2 -> 11
    # k=4: i=2 -> 14
    # k=5: i=2 -> 17
    
    # Schoolbook for 1x4x6.
    # 24 terms.
    for j in range(4):
        for k in range(6):
            full_u = [0] * (n * m)
            full_v = [0] * (m * p)
            full_w = [0] * (p * n)
            
            a_idx = 8 + j # Row 2, col j
            b_idx = j * 6 + k # Row j, col k
            c_idx = k * n + 2 # Col k, row 2
            
            full_u[a_idx] = 1
            full_v[b_idx] = 1
            full_w[c_idx] = 1
            
            U.append(full_u)
            V.append(full_v)
            W.append(full_w)
            
    return {"U": U, "V": V, "W": W}
