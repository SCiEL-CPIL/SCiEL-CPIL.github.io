import numpy as np

def construct():
    n, m, p = 2, 4, 6
    
    # Rank 42 decomposition using Strassen on 2x2 blocks
    # A is 2x4, split into A1 (cols 0,1) and A2 (cols 2,3)
    # B is 4x6, split into B_top (rows 0,1) and B_bot (rows 2,3)
    # B_top split into B_T0, B_T1, B_T2 (cols 0,1; 2,3; 4,5)
    # B_bot split into B_B0, B_B1, B_B2 (cols 0,1; 2,3; 4,5)
    # C is 2x6, split into C0, C1, C2 (cols 0,1; 2,3; 4,5)
    # Cb = A1 * B_Tb + A2 * B_Bb
    
    # Strassen terms for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Coefficients for U (A indices 00, 01, 10, 11)
    # Coefficients for V (B indices 00, 01, 10, 11)
    # Coefficients for W (C indices 00, 01, 10, 11)
    
    # Local indices for 2x2:
    # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Strassen vectors (u, v, w)
    # u is 4x7, v is 4x7, w is 4x7
    # w corresponds to C coefficients for each M_r
    
    U_strassen = np.array([
        [1, 0, 1, 1, 1, 1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1], # A01
        [0, 1, 0, 0, 0, 1, 0], # A10
        [1, 1, 0, 1, 0, 0, -1] # A11
    ], dtype=float)
    
    V_strassen = np.array([
        [1, 1, 0, -1, 0, 1, 0], # B00
        [1, 0, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ], dtype=float)
    
    W_strassen = np.array([
        [1, 0, 0, 1, 0, 0, 0], # C00 = M1 + M4 - M5 + M7 -> Wait, check Strassen formulas
        [0, 0, 1, 0, 1, 0, 0], # C01 = M3 + M5
        [0, 1, 0, 1, 0, 0, 0], # C10 = M2 + M4
        [1, -1, 1, 0, 0, 1, 0] # C11 = M1 - M2 + M3 + M6
    ], dtype=float)
    
    # Let's re-verify W_strassen from standard Strassen:
    # C11 (00) = M1 + M4 - M5 + M7
    # C12 (01) = M3 + M5
    # C21 (10) = M2 + M4
    # C22 (11) = M1 - M2 + M3 + M6
    
    # My W_strassen above:
    # Col 0 (M1): C00=1, C11=1. Correct.
    # Col 1 (M2): C10=1, C11=-1. Correct.
    # Col 2 (M3): C01=1, C11=1. Correct.
    # Col 3 (M4): C00=1, C10=1. Correct.
    # Col 4 (M5): C01=1, C00=-1. Correct.
    # Col 5 (M6): C11=1. Correct.
    # Col 6 (M7): C00=1. Correct.
    
    # Wait, C00 = M1 + M4 - M5 + M7.
    # W[0,0]=1 (M1), W[0,3]=1 (M4), W[0,4]=-1 (M5), W[0,6]=1 (M7).
    # My array:
    # Row 0 (C00): [1, 0, 0, 1, 0, 0, 0] -> Missing -M5 and +M7?
    # Ah, the array is indexed by [C_index, M_index].
    # So W_strassen[0, 4] should be -1. W_strassen[0, 6] should be 1.
    
    W_strassen = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ], dtype=float)
    
    # Now build global U, V, W
    rank = 42
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for b in range(3): # Block index 0, 1, 2
        # B columns for this block: 2*b, 2*b+1
        k_start = 2 * b
        
        for s in range(2): # 0 for A1/B_T, 1 for A2/B_B
            # A columns: if s=0, 0,1. if s=1, 2,3
            j_start = 2 * s
            # B rows: if s=0, 0,1. if s=1, 2,3
            j_row_start = 2 * s
            
            for r in range(7): # Strassen term index
                # Update U
                for i_local in range(2):
                    for j_local in range(2):
                        val = U_strassen[i_local*2 + j_local, r]
                        if val != 0:
                            i_global = i_local
                            j_global = j_start + j_local
                            U[i_global*m + j_global, col] += val
                            
                # Update V
                for j_local in range(2):
                    for k_local in range(2):
                        val = V_strassen[j_local*2 + k_local, r]
                        if val != 0:
                            j_global = j_row_start + j_local
                            k_global = k_start + k_local
                            V[j_global*p + k_global, col] += val
                            
                # Update W
                for i_local in range(2):
                    for k_local in range(2):
                        val = W_strassen[i_local*2 + k_local, r]
                        if val != 0:
                            i_global = i_local
                            k_global = k_start + k_local
                            W[k_global*n + i_global, col] += val
                            
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
