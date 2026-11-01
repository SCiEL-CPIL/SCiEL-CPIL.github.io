import numpy as np
import time

def construct():
    # Deadline guard
    start_time = time.time()
    
    n, m, p = 2, 4, 5
    
    # Rank 36 decomposition
    # A (2x4) split into A1 (2x2), A2 (2x2)
    # B (4x5) split into B_top (2x5), B_bot (2x5)
    # C = A1 @ B_top + A2 @ B_bot
    
    # We need a decomposition for 2x2 @ 2x5 with rank 18.
    # Split 2x5 into 2x2, 2x2, 2x1.
    # Term 1: Strassen on 2x2 (Rank 7)
    # Term 2: Strassen on 2x2 (Rank 7)
    # Term 3: Schoolbook on 2x1 (Rank 4)
    
    total_rank = 36
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    idx = 0
    
    # Helper to add Strassen decomposition for 2x2 X 2x2
    # X is at U[row_offset : row_offset+2, :], V[row_offset_v : row_offset_v+2, :]
    # But U is flattened 2x4. A1 is rows 0,1. A2 is rows 2,3?
    # A is 2x4. Rows 0,1. Cols 0,1 for A1. Cols 2,3 for A2.
    # U has shape (8, rank).
    # U[0,0]=A1[0,0], U[1,0]=A1[0,1], U[2,0]=A1[1,0], U[3,0]=A1[1,1]
    # Wait, standard indexing U[i*m+j, k].
    # A1[0,0] is U[0*4+0, k] = U[0,k]
    # A1[0,1] is U[0*4+1, k] = U[1,k]
    # A1[1,0] is U[1*4+0, k] = U[4,k]
    # A1[1,1] is U[1*4+1, k] = U[5,k]
    
    def add_strassen_2x2(u_off, v_off, w_off, rank_offset):
        # u_off: offset in U for the 2x2 block (row index in 2x4? No, flat index)
        # v_off: offset in V for the 2x2 block (row index in 2x5? No, flat index)
        # w_off: offset in W for the 2x2 block (col index in 2x5? No, flat index k*n+i)
        
        # Strassen:
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b12-b22)
        # M4 = a22(b21-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        
        # U contributions (A part)
        # V contributions (B part)
        # W contributions (C part)
        
        # A1 is U[u_off, u_off+1, u_off+4, u_off+5] ?
        # Let's assume u_off is the flat index of A[0,0] of the block.
        # For A1, u_off=0. A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[4], A[1,1]=U[5].
        # For A2, u_off=2? A[0,2]=U[2], A[0,3]=U[3], A[1,2]=U[6], A[1,3]=U[7].
        
        # We need to map the 4 values of the 2x2 block to specific flat indices.
        # Let a_idx = [u_idx0, u_idx1, u_idx4, u_idx5] for A1
        # And v_idx = [v_idx0, v_idx1, v_idx2, v_idx3] for B1 (2x2 block)
        # And w_idx = [w_idx0, w_idx1, w_idx2, w_idx3] for C1 (2x2 block)
        
        # This helper is too complex to generalize with offsets easily without explicit lists.
        # I will hardcode the 18 terms for Side 1 and Side 2.
        
        pass

    # Manual Construction for Side 1 (A1 * B1)
    # A1 indices in U: [0, 1, 4, 5] corresponding to [0,0; 0,1; 1,0; 1,1]
    # B1 indices in V: [0, 5, 1, 6, 2, 7, 3, 8, 4, 9] ? 
    # B is 4x5. B1 is top 2 rows, cols 0-1 (for first 2x2), 2-3 (second 2x2), 4 (1x1? No, 2x1)
    # B1 (2x5) flat indices in V:
    # Row 0 of B1 is Row 0 of B: indices 0,1,2,3,4
    # Row 1 of B1 is Row 1 of B: indices 5,6,7,8,9
    # First 2x2 block of B1: cols 0,1. Indices: 0,1,5,6.
    # Second 2x2 block of B1: cols 2,3. Indices: 2,3,7,8.
    # Last 2x1 block of B1: col 4. Indices: 4, 9.
    
    # C1 is top 2 rows of C. C is 2x5.
    # C indices in W (k*n+i, n=2):
    # C[0,0] -> W[0] (k=0, i=0)
    # C[0,1] -> W[2] (k=1, i=0)
    # C[1,0] -> W[1] (k=0, i=1)
    # C[1,1] -> W[3] (k=1, i=1)
    # C[0,2] -> W[4]
    # C[1,2] -> W[5]
    # C[0,3] -> W[6]
    # C[1,3] -> W[7]
    # C[0,4] -> W[8]
    # C[1,4] -> W[9]
    
    # First 2x2 of C1: [0,2,1,3] (C00, C01, C10, C11)
    # Second 2x2 of C1: [4,6,5,7]
    # Last 2x1 of C1: [8,9]
    
    # Side 2 (A2 * B2)
    # A2 indices in U: [2, 3, 6, 7]
    # B2 indices in V:
    # B2 is bottom 2 rows of B.
    # Row 2 of B: indices 10,11,12,13,14
    # Row 3 of B: indices 15,16,17,18,19
    # First 2x2: [10,11,15,16]
    # Second 2x2: [12,13,17,18]
    # Last 2x1: [14,19]
    
    # C2 is bottom 2 rows of C? No, C is 2x5.
    # Wait, C = A1 B1 + A2 B2.
    # A1 is 2x2, B1 is 2x5. Result 2x5.
    # A2 is 2x2, B2 is 2x5. Result 2x5.
    # C is 2x5.
    # So C1 and C2 are both 2x5 and added together.
    # W indices for Side 2 are the same as Side 1: [0..9].
    
    def add_strassen_term(u_list, v_list, w_list, signs_u, signs_v, signs_w, rank_offset):
        # u_list: 4 indices
        # v_list: 4 indices
        # w_list: 4 indices (C00, C01, C10, C11)
        # signs_u: 4 values for U
        # signs_v: 4 values for V
        # signs_w: 4 values for W
        pass

    # It's safer to just fill the arrays directly for the 36 terms.
    
    terms = []
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        U[u_idx, idx] = u_val
        V[v_idx, idx] = v_val
        W[w_idx, idx] = w_val
        nonlocal idx
        idx += 1

    # Side 1
    # A1: u=[0,1,4,5] (a00, a01, a10, a11)
    # B1_1: v=[0,1,5,6] (b00, b01, b10, b11)
    # C1_1: w=[0,2,1,3] (c00, c01, c10, c11)
    
    # Strassen 1 on A1, B1_1
    # M1
    add_term(0, 1, 0, 1, 0, 1); add_term(5, 1, 0, 1, 0, 1) # a00,a11
    add_term(0, 1, 1, 1, 2, 1); add_term(5, 1, 1, 1, 2, 1) # b00,b01
    # ... This is getting messy. I will use a generic loop for the 7 terms.
    
    def execute_strassen(u_base, v_base, w_base, block_u, block_v, block_w):
        # block_u: [u00, u01, u10, u11]
        # block_v: [v00, v01, v10, v11]
        # block_w: [w00, w01, w10, w11]
        
        # M1 = (a00+a11)(b00+b11)
        add_term(block_u[0], 1, block_v[0], 1, block_w[0], 1)
        add_term(block_u[0], 1, block_v[1], 1, block_w[1], 1)
        add_term(block_u[0], 1, block_v[2], -1, block_w[0], 1)
        add_term(block_u[0], 1, block_v[3], -1, block_w[1], 1)
        add_term(block_u[3], 1, block_v[0], 1, block_w[0], 1)
        add_term(block_u[3], 1, block_v[1], 1, block_w[1], 1)
        add_term(block_u[3], 1, block_v[2], -1, block_w[0], 1)
        add_term(block_u[3], 1, block_v[3], -1, block_w[1], 1)
        
        # This is not rank 7, this is rank 8 (2x2 schoolbook).
        # Strassen requires specific combinations.
        
        # I will implement the 7 terms explicitly.
        
        # Term 1: (a00+a11)(b00+b11)
        U[block_u[0], idx] = 1; V[block_v[0], idx] = 1; W[block_w[0], idx] = 1
        U[block_u[0], idx] = 1; V[block_v[1], idx] = 1; W[block_w[1], idx] = 1
        U[block_u[0], idx] = 1; V[block_v[2], idx] = -1; W[block_w[0], idx] = 1
        U[block_u[0], idx] = 1; V[block_v[3], idx] = -1; W[block_w[1], idx] = 1
        U[block_u[3], idx] = 1; V[block_v[0], idx] = 1; W[block_w[0], idx] = 1
        U[block_u[3], idx] = 1; V[block_v[1], idx] = 1; W[block_w[1], idx] = 1
        U[block_u[3], idx] = 1; V[block_v[2], idx] = -1; W[block_w[0], idx] = 1
        U[block_u[3], idx] = 1; V[block_v[3], idx] = -1; W[block_w[1], idx] = 1
        idx += 1
        
        # Term 2: (a10+a11)b00
        U[block_u[2], idx] = 1; V[block_v[0], idx] = 1; W[block_w[0], idx] = 1
        U[block_u[3], idx] = 1; V[block_v[0], idx] = 1; W[block_w[0], idx] = 1
        idx += 1
        # ... etc
        
    # This is too verbose. I will return the rank 40 solution if I cannot fit the rank 36 construction in time.
    # But the rank 36 is the target.
    
    # Given the complexity and risk, and the time limit, I will return the Rank 40 solution but with a different structure (e.g. using lists) to ensure validity.
    # However, the directive asks for a stronger algorithm.
    # I will stick to the Rank 36 logic but implement it carefully.
    
    pass
