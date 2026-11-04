import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a Strassen-based decomposition.
    # Split A (2x4) into A1 (2x2) and A2 (2x2)
    # Split B (4x8) into B1 (2x8) and B2 (2x8)
    # B1 split into 4 blocks B10, B11, B12, B13 (2x2)
    # B2 split into 4 blocks B20, B21, B22, B23 (2x2)
    # C = A1*B1 + A2*B2
    
    # Strassen coefficients for 2x2 * 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h) -> C11, C22
    # M2 = (c+d)e   -> C21, C22
    # M3 = a(f-h)   -> C12, C22
    # M4 = d(g-e)   -> C11, C21
    # M5 = (a+b)h   -> C11, C12
    # M6 = (c-a)(e+f) -> C22
    # M7 = (b-d)(g+h) -> C11
    
    # Local indices for A (2x2): 0=a, 1=b, 2=c, 3=d
    # Local indices for B (2x2): 0=e, 1=f, 2=g, 3=h
    # Local indices for C (2x2): 0=C11, 1=C12, 2=C21, 3=C22
    
    strassen_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    rank = 0
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a term
    def add_term(u_local, v_local, w_local, a_offset, b_offset, c_offset):
        nonlocal rank
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        
        # Map local A indices to global
        # A is 2x4. A1 is cols 0,1. A2 is cols 2,3.
        # Local 0 -> global a_offset + 0
        # Local 1 -> global a_offset + 1
        # Local 2 -> global a_offset + 2
        # Local 3 -> global a_offset + 3
        # Note: a_offset is the row-major index of the top-left of the 2x2 block in A.
        # For A1 (top-left 2x2 of 2x4), a_offset = 0.
        # For A2 (top-right 2x2 of 2x4), a_offset = 2.
        
        for i in range(4):
            if u_local[i] != 0:
                u_vec[a_offset + i] = u_local[i]
                
        # Map local B indices to global
        # B is 4x8. 
        # B1 is rows 0,1. B2 is rows 2,3.
        # B10 is cols 0,1 of B1.
        # B11 is cols 2,3 of B1.
        # ...
        # b_offset is the row-major index of the top-left of the 2x2 block in B.
        
        for i in range(4):
            if v_local[i] != 0:
                v_vec[b_offset + i] = v_local[i]
                
        # Map local C indices to global
        # C is 8x2.
        # C11 is rows 0,1 of C, cols 0,1? No.
        # C is n x p = 2 x 8.
        # C11 (from A1 B10) corresponds to C rows 0,1 and cols 0,1.
        # C12 (from A1 B11) corresponds to C rows 0,1 and cols 2,3.
        # C13 (from A1 B12) corresponds to C rows 0,1 and cols 4,5.
        # C14 (from A1 B13) corresponds to C rows 0,1 and cols 6,7.
        # C21 (from A2 B20) corresponds to C rows 0,1 and cols 0,1. (Added to C11)
        # ...
        
        # c_offset is the row-major index of the top-left of the 2x2 block in C.
        # Wait, C is 2x8. Row-major index of C[i, j] is i * 8 + j.
        # For a 2x2 block at row 0, col c_start:
        # Indices: c_start, c_start+1, c_start+8, c_start+9.
        # Local 0 (C11) -> global c_offset + 0
        # Local 1 (C12) -> global c_offset + 1
        # Local 2 (C21) -> global c_offset + 8
        # Local 3 (C22) -> global c_offset + 9
        
        for i in range(4):
            if w_local[i] != 0:
                w_vec[c_offset + i] = w_local[i]
                
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        rank += 1

    # A1 is at offset 0 in A (2x4). Indices 0,1,2,3.
    # A2 is at offset 2 in A (2x4). Indices 2,3,4,5? 
    # A is 2x4. Row 0: 0,1,2,3. Row 1: 4,5,6,7.
    # A1 (cols 0,1): 0,1,4,5.
    # A2 (cols 2,3): 2,3,6,7.
    # My local mapping assumed contiguous 0,1,2,3. 
    # I need to map local indices to specific global indices.
    
    # Let's refine the mapping.
    # A1 local: 0->0, 1->1, 2->4, 3->5
    # A2 local: 0->2, 1->3, 2->6, 3->7
    
    # B1 (rows 0,1) of B (4x8).
    # B10 (cols 0,1): 0,1,8,9
    # B11 (cols 2,3): 2,3,10,11
    # B12 (cols 4,5): 4,5,12,13
    # B13 (cols 6,7): 6,7,14,15
    # B2 (rows 2,3) of B (4x8).
    # B20 (cols 0,1): 16,17,24,25
    # B21 (cols 2,3): 18,19,26,27
    # B22 (cols 4,5): 20,21,28,29
    # B23 (cols 6,7): 22,23,30,31
    
    # C (2x8).
    # C11 (from A1 B10): C[0,0], C[0,1], C[1,0], C[1,1] -> 0,1,8,9
    # C12 (from A1 B11): C[0,2], C[0,3], C[1,2], C[1,3] -> 2,3,10,11
    # C13 (from A1 B12): C[0,4], C[0,5], C[1,4], C[1,5] -> 4,5,12,13
    # C14 (from A1 B13): C[0,6], C[0,7], C[1,6], C[1,7] -> 6,7,14,15
    # C21 (from A2 B20): Same as C11 -> 0,1,8,9
    # C22 (from A2 B21): Same as C12 -> 2,3,10,11
    # C23 (from A2 B22): Same as C13 -> 4,5,12,13
    # C24 (from A2 B23): Same as C14 -> 6,7,14,15
    
    # We will iterate through the 8 blocks.
    
    # Block 0: A1 * B10 -> C11
    # A1 map: [0, 1, 4, 5]
    # B10 map: [0, 1, 8, 9]
    # C11 map: [0, 1, 8, 9]
    
    # Block 1: A1 * B11 -> C12
    # A1 map: [0, 1, 4, 5]
    # B11 map: [2, 3, 10, 11]
    # C12 map: [2, 3, 10, 11]
    
    # Block 2: A1 * B12 -> C13
    # A1 map: [0, 1, 4, 5]
    # B12 map: [4, 5, 12, 13]
    # C13 map: [4, 5, 12, 13]
    
    # Block 3: A1 * B13 -> C14
    # A1 map: [0, 1, 4, 5]
    # B13 map: [6, 7, 14, 15]
    # C14 map: [6, 7, 14, 15]
    
    # Block 4: A2 * B20 -> C11
    # A2 map: [2, 3, 6, 7]
    # B20 map: [16, 17, 24, 25]
    # C11 map: [0, 1, 8, 9]
    
    # Block 5: A2 * B21 -> C12
    # A2 map: [2, 3, 6, 7]
    # B21 map: [18, 19, 26, 27]
    # C12 map: [2, 3, 10, 11]
    
    # Block 6: A2 * B22 -> C13
    # A2 map: [2, 3, 6, 7]
    # B22 map: [20, 21, 28, 29]
    # C13 map: [4, 5, 12, 13]
    
    # Block 7: A2 * B23 -> C14
    # A2 map: [2, 3, 6, 7]
    # B23 map: [22, 23, 30, 31]
    # C14 map: [6, 7, 14, 15]
    
    blocks = [
        ([0, 1, 4, 5], [0, 1, 8, 9], [0, 1, 8, 9]),
        ([0, 1, 4, 5], [2, 3, 10, 11], [2, 3, 10, 11]),
        ([0, 1, 4, 5], [4, 5, 12, 13], [4, 5, 12, 13]),
        ([0, 1, 4, 5], [6, 7, 14, 15], [6, 7, 14, 15]),
        ([2, 3, 6, 7], [16, 17, 24, 25], [0, 1, 8, 9]),
        ([2, 3, 6, 7], [18, 19, 26, 27], [2, 3, 10, 11]),
        ([2, 3, 6, 7], [20, 21, 28, 29], [4, 5, 12, 13]),
        ([2, 3, 6, 7], [22, 23, 30, 31], [6, 7, 14, 15]),
    ]
    
    for a_map, b_map, c_map in blocks:
        for u_l, v_l, w_l in strassen_terms:
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            
            for i in range(4):
                if u_l[i] != 0:
                    u_vec[a_map[i]] = u_l[i]
                if v_l[i] != 0:
                    v_vec[b_map[i]] = v_l[i]
                if w_l[i] != 0:
                    w_vec[c_map[i]] = w_l[i]
                    
            U_cols.append(u_vec)
            V_cols.append(v_vec)
            W_cols.append(w_vec)
            
    rank = len(U_cols)
    
    # Convert to matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for i in range(n * m):
            U[i][c] = U_cols[c][i]
        for i in range(m * p):
            V[i][c] = V_cols[c][i]
        for i in range(p * n):
            W[i][c] = W_cols[c][i]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
