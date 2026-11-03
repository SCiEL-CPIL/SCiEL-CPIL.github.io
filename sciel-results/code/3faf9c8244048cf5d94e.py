import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # We decompose the 2x4 x 4x5 multiplication into two 2x2 x 2x5 multiplications.
    # A = [A1, A2] where A1, A2 are 2x2.
    # B = [B1; B2] where B1, B2 are 2x5.
    # C = A1*B1 + A2*B2.
    # We solve two independent 2x2 x 2x5 problems.
    
    # Helper: Construct Strassen decomposition for 2x2 x 2x2
    def strassen_2x2x2():
        # Returns U, V, W for 2x2 x 2x2 with rank 7
        # Indices: A is 2x2 (idx 0-3), B is 2x2 (idx 0-3), C is 2x2 (idx 0-3)
        # U shape (4, 7), V shape (4, 7), W shape (4, 7)
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00 + A11)(B00 + B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10 + A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1 # Wait, W maps to C indices.
        # C00 = M1 - M4 - M5 + M7
        # C01 = M5 + M6
        # C10 = M2 + M3
        # C11 = M1 - M2 - M3 + M6
        
        # Let's use the standard Strassen coefficients for W (C side)
        # C00 = M1 - M4 - M5 + M7
        W[0, 0] = 1
        W[0, 3] = -1
        W[0, 4] = -1
        W[0, 6] = 1
        
        # C01 = M5 + M6
        W[1, 4] = 1
        W[1, 5] = 1
        
        # C10 = M2 + M3
        W[2, 1] = 1
        W[2, 2] = 1
        
        # C11 = M1 - M2 - M3 + M6
        W[3, 0] = 1
        W[3, 1] = -1
        W[3, 2] = -1
        W[3, 5] = 1
        
        # Re-define U and V carefully
        # M1 = (A00+A11)(B00+B11)
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        
        # M2 = (A10+A11)B00
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        
        # M3 = A00(B10-B11)
        U[0,2]=1
        V[2,2]=1; V[3,2]=-1
        
        # M4 = A11(B01-B00)
        U[3,3]=1
        V[1,3]=1; V[0,3]=-1
        
        # M5 = (A00+A01)B11
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        
        # M6 = (A10-A00)(B00+B01)
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        
        # M7 = (A01-A11)(B10+B11)
        U[1,6]=1; U[3,6]=-1
        V[2,6]=1; V[3,6]=1
        
        return U, V, W

    # Helper: Construct Standard decomposition for 2x2 x 2x1 (Rank 4)
    def standard_2x2x1():
        # A (2x2), B (2x1), C (2x1)
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        
        # M1 = A00 B0
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        # M2 = A01 B1
        U[1,1]=1; V[1,1]=1; W[0,1]=1
        # M3 = A10 B0
        U[2,2]=1; V[0,2]=1; W[1,2]=1
        # M4 = A11 B1
        U[3,3]=1; V[1,3]=1; W[1,3]=1
        
        return U, V, W

    # Helper: Construct Standard decomposition for 2x2 x 2x2 (Rank 4) -- Actually Strassen is better, but for consistency in block logic
    # We will use Strassen for 2x2x2 blocks.

    # We need to assemble the global U, V, W for 2x4 x 4x5.
    # Global U: (8, Rank)
    # Global V: (20, Rank)
    # Global W: (10, Rank)
    
    # The two sub-problems are:
    # 1. A1 (2x2) * B1 (2x5) -> C1 (2x5)
    # 2. A2 (2x2) * B2 (2x5) -> C2 (2x5)
    
    # For each 2x2 x 2x5:
    # Split B (2x5) into B_left (2x2) and B_right (2x3).
    # B_right is split into B_col3 (2x1), B_col4 (2x1), B_col5 (2x1).
    # We use Strassen for B_left (Rank 7).
    # We use Standard for each of the 3 columns (Rank 4 each).
    # Total rank per sub-problem: 7 + 4 + 4 + 4 = 19.
    # Total global rank: 19 + 19 = 38.
    
    total_rank = 38
    U = np.zeros((8, total_rank))
    V = np.zeros((20, total_rank))
    W = np.zeros((10, total_rank))
    
    col_offset = 0
    
    for sub in range(2):
        # Sub-problem indices
        # A_sub is rows 0-3 of A (if sub=0) or 4-7 (if sub=1)?
        # A is 2x4. A = [A1, A2].
        # A1 is cols 0-1 of A. A2 is cols 2-3 of A.
        # In U (shape 8xRank), rows correspond to A indices.
        # A index i*m+j. n=2, m=4.
        # A1: i=0,1; j=0,1. Indices: 0,1,4,5.
        # A2: i=0,1; j=2,3. Indices: 2,3,6,7.
        
        a_rows = [0, 1, 4, 5] if sub == 0 else [2, 3, 6, 7]
        
        # B_sub is rows 0-9 of B (if sub=0) or 10-19 (if sub=1)?
        # B is 4x5. B = [B1; B2].
        # B1 is rows 0-1 of B. B2 is rows 2-3 of B.
        # B index j*p+k. m=4, p=5.
        # B1: j=0,1; k=0..4. Indices: 0..4, 5..9.
        # B2: j=2,3; k=0..4. Indices: 10..14, 15..19.
        
        b_rows = list(range(0, 10)) if sub == 0 else list(range(10, 20))
        
        # C_sub is rows 0-4 of C (if sub=0) or 5-9 (if sub=1)?
        # C is 2x5. C = C1 + C2? No, C = A1 B1 + A2 B2.
        # C1 is 2x5. C2 is 2x5.
        # C index k*n+i. p=5, n=2.
        # C1 contributes to all 10 entries.
        # C2 contributes to all 10 entries.
        # So W rows are shared.
        
        c_rows = list(range(0, 10))
        
        # 1. Strassen part for B_left (cols 0,1 of B_sub)
        # B_left indices in global V:
        # If sub=0: B1 cols 0,1 -> V indices 0,1,5,6
        # If sub=1: B2 cols 0,1 -> V indices 10,11,15,16
        
        b_left_local = [0, 1, 5, 6] if sub == 0 else [10, 11, 15, 16]
        
        U_s, V_s, W_s = strassen_2x2x2()
        
        for k in range(7):
            # Map U_s rows (0-3) to global U rows (a_rows)
            for r in range(4):
                U[a_rows[r], col_offset + k] = U_s[r, k]
            
            # Map V_s rows (0-3) to global V rows (b_left_local)
            for r in range(4):
                V[b_left_local[r], col_offset + k] = V_s[r, k]
            
            # Map W_s rows (0-3) to global W rows (c_rows)
            # W_s is for C (2x2). C indices 0,1,2,3 correspond to global C indices?
            # Global C index k*n+i.
            # C1 (from A1 B1) is 2x2 (cols 0,1).
            # Indices:
            # C[0,0] -> 0*2+0 = 0
            # C[0,1] -> 1*2+0 = 2
            # C[1,0] -> 0*2+1 = 1
            # C[1,1] -> 1*2+1 = 3
            # So W_s[0] -> W[0], W_s[1] -> W[2], W_s[2] -> W[1], W_s[3] -> W[3]
            w_map = [0, 2, 1, 3]
            for r in range(4):
                W[c_rows[w_map[r]], col_offset + k] = W_s[r, k]
        
        col_offset += 7
        
        # 2. Standard parts for B_right (cols 2,3,4 of B_sub)
        for col_idx in range(3):
            # B_col indices in global V
            # If sub=0: B1 col 2,3,4 -> V indices 2, 7, 3, 8, 4, 9
            # If sub=1: B2 col 2,3,4 -> V indices 12, 17, 13, 18, 14, 19
            
            if sub == 0:
                b_col_local = [2 + col_idx, 7 + col_idx]
            else:
                b_col_local = [12 + col_idx, 17 + col_idx]
                
            U_st, V_st, W_st = standard_2x2x1()
            
            for k in range(4):
                # Map U_st rows (0-3) to global U rows (a_rows)
                for r in range(4):
                    U[a_rows[r], col_offset + k] = U_st[r, k]
                
                # Map V_st rows (0-1) to global V rows (b_col_local)
                # V_st is 2x1. Rows 0,1.
                V[b_col_local[0], col_offset + k] = V_st[0, k]
                V[b_col_local[1], col_offset + k] = V_st[1, k]
                
                # Map W_st rows (0-1) to global W rows (c_rows)
                # W_st is 2x1.
                # C col index is 2 + col_idx.
                # C indices:
                # C[0, c] -> c*2 + 0
                # C[1, c] -> c*2 + 1
                c_col = 2 + col_idx
                w_idx_0 = c_col * 2 + 0
                w_idx_1 = c_col * 2 + 1
                
                W[c_rows[w_idx_0], col_offset + k] = W_st[0, k]
                W[c_rows[w_idx_1], col_offset + k] = W_st[1, k]
            
            col_offset += 4
            
    # Convert to list of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
