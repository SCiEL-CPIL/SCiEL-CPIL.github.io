import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using 4 Strassen blocks
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    
    # U vectors for A (indices 0,1,2,3 correspond to a,b,c,d)
    # V vectors for B (indices 0,1,2,3 correspond to e,f,g,h)
    # W vectors for C (indices 0,1,2,3 correspond to C00, C01, C10, C11)
    
    # Note: W index mapping in local 2x2 output:
    # C00 -> 0, C01 -> 1, C10 -> 2, C11 -> 3
    # But task W index is k*n + i.
    # Local k=0, i=0 -> 0. Local k=1, i=0 -> 2.
    # Local k=0, i=1 -> 1. Local k=1, i=1 -> 3.
    # So local W indices:
    # C00 (i=0,k=0) -> 0
    # C01 (i=0,k=1) -> 2
    # C10 (i=1,k=0) -> 1
    # C11 (i=1,k=1) -> 3
    
    strassen_terms = [
        # (U_local, V_local, W_local_coeffs)
        # U_local: [a,b,c,d], V_local: [e,f,g,h]
        # W_local: [C00, C01, C10, C11] contributions
        
        # P1 = (a+d)(e+h)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # P2 = (c+d)e
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # C10 += 1, C11 -= 1? 
        # Wait, Strassen output formulas:
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # Let's re-verify P2 contribution.
        # P2 is in C10 (+1) and C11 (-1).
        # So W_local for P2 should be [0, 0, 1, -1].
        
        # P3 = a(f-h)
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # C01 += 1, C11 += 1
        
        # P4 = d(g-e)
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # C00 += 1, C10 += 1
        
        # P5 = (a+b)h
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # C00 -= 1, C01 += 1
        
        # P6 = (c-a)(e+f)
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # C11 += 1
        
        # P7 = (b-d)(g+h)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # C00 += 1
    ]
    
    # Correct W coefficients based on formulas:
    # C00: P1(+1), P4(+1), P5(-1), P7(+1)
    # C01: P3(+1), P5(+1)
    # C10: P2(+1), P4(+1)
    # C11: P1(+1), P2(-1), P3(+1), P6(+1)
    
    # My previous list:
    # P1: [1, 0, 0, 1] -> C00+1, C11+1. Correct.
    # P2: [0, 0, 1, -1] -> C10+1, C11-1. Correct.
    # P3: [0, 1, 0, 1] -> C01+1, C11+1. Correct.
    # P4: [1, 0, 1, 0] -> C00+1, C10+1. Correct.
    # P5: [-1, 1, 0, 0] -> C00-1, C01+1. Correct.
    # P6: [0, 0, 0, 1] -> C11+1. Correct.
    # P7: [1, 0, 0, 0] -> C00+1. Correct.
    
    # Define index mappings
    # A indices: i*m + j. m=4.
    # A1 (cols 0,1): 0,1 (row 0), 4,5 (row 1).
    # A2 (cols 2,3): 2,3 (row 0), 6,7 (row 1).
    
    # B indices: j*p + k. p=4.
    # B1L (rows 0,1, cols 0,1): 0,1, 4,5.
    # B2L (rows 2,3, cols 0,1): 8,9, 12,13.
    # B1R (rows 0,1, cols 2,3): 2,3, 6,7.
    # B2R (rows 2,3, cols 2,3): 10,11, 14,15.
    
    # W indices: k*n + i. n=2.
    # CL (k=0,1):
    #   Local C00 (i=0,k=0) -> 0
    #   Local C01 (i=0,k=1) -> 2
    #   Local C10 (i=1,k=0) -> 1
    #   Local C11 (i=1,k=1) -> 3
    # CR (k=2,3):
    #   Local C00 (i=0,k=2) -> 4
    #   Local C01 (i=0,k=3) -> 6
    #   Local C10 (i=1,k=2) -> 5
    #   Local C11 (i=1,k=3) -> 7
    
    def add_block(U_indices, V_indices, W_offset, col_start):
        for i, (u_loc, v_loc, w_loc) in enumerate(strassen_terms):
            col = col_start + i
            # U
            for idx, val in enumerate(u_loc):
                if val != 0:
                    U[U_indices[idx]][col] = val
            # V
            for idx, val in enumerate(v_loc):
                if val != 0:
                    V[V_indices[idx]][col] = val
            # W
            # w_loc is [C00, C01, C10, C11]
            # Map to global W indices
            # C00 -> W_offset[0]
            # C01 -> W_offset[1]
            # C10 -> W_offset[2]
            # C11 -> W_offset[3]
            for idx, val in enumerate(w_loc):
                if val != 0:
                    W[W_offset[idx]][col] = val

    # Block 1: A1, B1L -> CL
    U_idx_1 = [0, 1, 4, 5]
    V_idx_1 = [0, 1, 4, 5]
    W_off_1 = [0, 2, 1, 3]
    add_block(U_idx_1, V_idx_1, W_off_1, 0)
    
    # Block 2: A2, B2L -> CL
    U_idx_2 = [2, 3, 6, 7]
    V_idx_2 = [8, 9, 12, 13]
    W_off_2 = [0, 2, 1, 3]
    add_block(U_idx_2, V_idx_2, W_off_2, 7)
    
    # Block 3: A1, B1R -> CR
    U_idx_3 = [0, 1, 4, 5]
    V_idx_3 = [2, 3, 6, 7]
    W_off_3 = [4, 6, 5, 7]
    add_block(U_idx_3, V_idx_3, W_off_3, 14)
    
    # Block 4: A2, B2R -> CR
    U_idx_4 = [2, 3, 6, 7]
    V_idx_4 = [10, 11, 14, 15]
    W_off_4 = [4, 6, 5, 7]
    add_block(U_idx_4, V_idx_4, W_off_4, 21)
    
    return {"U": U, "V": V, "W": W}
