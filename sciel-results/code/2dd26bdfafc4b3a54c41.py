import numpy as np

def construct():
    n, m, p = 2, 2, 12
    
    # Strassen terms definition
    # Each term is a list of (coeff, A_idx, B_idx, C_idx)
    # But it's simpler to define the linear combinations for U, V, W directly.
    
    # Define the 7 Strassen multiplications for a 2x2x2 block
    # We will generate the contributions for one block, then replicate for 6 blocks.
    
    # Indices for A (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for B (2x2 sub-block): 0:(0,0), 1:(1,0) -> relative to block
    # Indices for C (2x2 sub-block): 0:(0,0), 1:(1,0) -> relative to block
    
    # Let's define the sparse entries for one 2x2x2 Strassen block.
    # Term 0: M1 = (A00+A11)(B00+B11)
    # U: A00(1), A11(1)
    # V: B00(1), B11(1)
    # W: C00(1), C11(1)
    
    # Term 1: M2 = (A10+A11)B00
    # U: A10(1), A11(1)
    # V: B00(1)
    # W: C10(1), C11(-1)
    
    # Term 2: M3 = A00(B01-B11)
    # U: A00(1)
    # V: B01(1), B11(-1)
    # W: C01(1), C11(1)
    
    # Term 3: M4 = A11(B10-B00)
    # U: A11(1)
    # V: B10(1), B00(-1)
    # W: C00(1), C10(1)
    
    # Term 4: M5 = (A00+A01)B11
    # U: A00(1), A01(1)
    # V: B11(1)
    # W: C00(-1), C01(1)
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    # U: A10(1), A00(-1)
    # V: B00(1), B01(1)
    # W: C11(1)
    
    # Term 6: M7 = (A01-A11)(B10+B11)
    # U: A01(1), A11(-1)
    # V: B10(1), B11(1)
    # W: C00(1)

    # We will store these as a list of tuples (U_entries, V_entries, W_entries)
    # where each entry is (index, coeff)
    
    strassen_block_terms = [
        # 0
        ([(0, 1), (3, 1)], [(0, 1), (1, 1)], [(0, 1), (3, 1)]),
        # 1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # 2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # 3
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # 4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # 5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # 6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    # Note: 
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B sub-block indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    #    Wait, my previous manual check used B00, B01, B10, B11.
    #    Let's map B indices carefully.
    #    B is 2x2. Row j, Col k.
    #    B00 is j=0, k=0 -> idx 0
    #    B01 is j=0, k=1 -> idx 1
    #    B10 is j=1, k=0 -> idx 2
    #    B11 is j=1, k=1 -> idx 3
    #
    #    C indices: Row i, Col k.
    #    C00 is i=0, k=0 -> idx 0
    #    C01 is i=0, k=1 -> idx 1
    #    C10 is i=1, k=0 -> idx 2
    #    C11 is i=1, k=1 -> idx 3
    
    # Let's re-verify Term 1: M2 = (A10+A11)B00
    # U: A10 (idx 2), A11 (idx 3). Correct.
    # V: B00 (idx 0). Correct.
    # W: C10 (idx 2), C11 (idx 3).
    # Formula: C10 = M2 + M4. C11 = M1 - M2 + M3 + M6.
    # So W for M2 should contribute +1 to C10 and -1 to C11.
    # My list: [(2, 1), (3, -1)]. Correct.

    # Let's re-verify Term 2: M3 = A00(B01-B11)
    # U: A00 (idx 0). Correct.
    # V: B01 (idx 1), B11 (idx 3). Correct.
    # W: C01 (idx 1), C11 (idx 3).
    # Formula: C01 = M3 + M5. C11 = M1 - M2 + M3 + M6.
    # So W for M3 should contribute +1 to C01 and +1 to C11.
    # My list: [(1, 1), (3, 1)]. Correct.

    # Let's re-verify Term 3: M4 = A11(B10-B00)
    # U: A11 (idx 3). Correct.
    # V: B10 (idx 2), B00 (idx 0). Correct.
    # W: C00 (idx 0), C10 (idx 2).
    # Formula: C00 = M1 + M4 - M5 + M7. C10 = M2 + M4.
    # So W for M4 should contribute +1 to C00 and +1 to C10.
    # My list: [(0, 1), (2, 1)]. Correct.

    # Let's re-verify Term 4: M5 = (A00+A01)B11
    # U: A00 (idx 0), A01 (idx 1). Correct.
    # V: B11 (idx 3). Correct.
    # W: C00 (idx 0), C01 (idx 1).
    # Formula: C00 = ... - M5. C01 = M3 + M5.
    # So W for M5 should contribute -1 to C00 and +1 to C01.
    # My list: [(0, -1), (1, 1)]. Correct.

    # Let's re-verify Term 5: M6 = (A10-A00)(B00+B01)
    # U: A10 (idx 2), A00 (idx 0). Correct.
    # V: B00 (idx 0), B01 (idx 1). Correct.
    # W: C11 (idx 3).
    # Formula: C11 = ... + M6.
    # So W for M6 should contribute +1 to C11.
    # My list: [(3, 1)]. Correct.

    # Let's re-verify Term 6: M7 = (A01-A11)(B10+B11)
    # U: A01 (idx 1), A11 (idx 3). Correct.
    # V: B10 (idx 2), B11 (idx 3). Correct.
    # W: C00 (idx 0).
    # Formula: C00 = ... + M7.
    # So W for M7 should contribute +1 to C00.
    # My list: [(0, 1)]. Correct.

    rank = 42
    U = np.zeros((4, rank), dtype=np.float64)
    V = np.zeros((24, rank), dtype=np.float64)
    W = np.zeros((24, rank), dtype=np.float64)
    
    # Map local block indices to global indices
    # Global U index: 2*i + j
    # Global V index: 12*j + k
    # Global W index: 2*k + i
    
    num_blocks = 6
    
    for b in range(num_blocks):
        # Block b handles columns 2b and 2b+1
        col_offset = 2 * b
        
        for s in range(7):
            r = b * 7 + s
            u_entries, v_entries, w_entries = strassen_block_terms[s]
            
            for idx, coeff in u_entries:
                # idx is 0..3 corresponding to A[i,j]
                # A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
                i = idx // 2
                j = idx % 2
                U[2*i + j, r] += coeff
                
            for idx, coeff in v_entries:
                # idx is 0..3 corresponding to B[j,k_local]
                # B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
                j = idx // 2
                k_local = idx % 2
                k_global = col_offset + k_local
                V[12*j + k_global, r] += coeff
                
            for idx, coeff in w_entries:
                # idx is 0..3 corresponding to C[i,k_local]
                # C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
                i = idx // 2
                k_local = idx % 2
                k_global = col_offset + k_local
                W[2*k_global + i, r] += coeff

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
