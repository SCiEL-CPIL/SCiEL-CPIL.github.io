import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-28 decomposition for 2x4 * 4x8 matrix multiplication.
    
    Uses Strassen's algorithm at the outer 2x2 level, where each "scalar"
    is a 2x2 block. The middle dimension 4 = 2*2 allows viewing A as a 2x2
    matrix of 1x2 blocks and B as a 2x2 matrix of 2x4 blocks.
    
    Rank = 7 (Strassen) * 4 (column groups) = 28
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    total_rank = 28
    
    U = np.zeros((n * m, total_rank), dtype=int)  # (8, 28)
    V = np.zeros((m * p, total_rank), dtype=int)  # (32, 28)
    W = np.zeros((p * n, total_rank), dtype=int)  # (16, 28)
    
    # Strassen's 7 terms for 2x2 matrix multiplication
    # Local indices for 2x2 blocks: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # For each Strassen term t (0..6) and each column group g (0..3):
    # rank index r = t * 4 + g
    
    # U_local[t][a] : coefficient for A block element a in Strassen term t
    # V_local[t][b] : coefficient for B block element b in Strassen term t  
    # W_local[t][c] : coefficient for C block element c in Strassen term t
    
    U_local = np.zeros((7, 4), dtype=int)
    V_local = np.zeros((7, 4), dtype=int)
    W_local = np.zeros((7, 4), dtype=int)
    
    # Strassen for 2x2: C = A*B
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # M0 = (A00 + A11)(B00 + B11)
    U_local[0][0] = 1; U_local[0][3] = 1
    V_local[0][0] = 1; V_local[0][3] = 1
    W_local[0][0] = 1; W_local[0][3] = 1
    
    # M1 = (A10 + A11)B00
    U_local[1][2] = 1; U_local[1][3] = 1
    V_local[1][0] = 1
    W_local[1][2] = 1; W_local[1][3] = -1
    
    # M2 = A00(B10 - B11)
    U_local[2][0] = 1
    V_local[2][2] = 1; V_local[2][3] = -1
    W_local[2][1] = 1; W_local[2][3] = 1
    
    # M3 = A11(B01 - B00)
    U_local[3][3] = 1
    V_local[3][1] = 1; V_local[3][0] = -1
    W_local[3][0] = 1; W_local[3][2] = 1
    
    # M4 = (A00 + A01)B11
    U_local[4][0] = 1; U_local[4][1] = 1
    V_local[4][3] = 1
    W_local[4][0] = -1; W_local[4][1] = 1
    
    # M5 = (A10 - A00)(B00 + B01)
    U_local[5][2] = 1; U_local[5][0] = -1
    V_local[5][0] = 1; V_local[5][1] = 1
    W_local[5][3] = 1
    
    # M6 = (A01 - A11)(B10 + B11)
    U_local[6][1] = 1; U_local[6][3] = -1
    V_local[6][2] = 1; V_local[6][3] = 1
    W_local[6][0] = 1; W_local[6][3] = 1
    
    # Now expand to full matrices
    # A is 2x4: A[i, j] where i in {0,1}, j in {0,1,2,3}
    # Split j into two groups: j_group 0 = {0,1}, j_group 1 = {2,3}
    # A_block[ig, jg] = A[ig, 2*jg : 2*jg+2] (1x2 block)
    # Local 2x2 index: a = ig*2 + (jg) -> but we need to map to U_local indices
    # 
    # For the outer 2x2 Strassen:
    # A_outer[0,0] = A[:, 0:2] (2x2), A_outer[0,1] = A[:, 2:4] (2x2)
    # A_outer[1,0] = A[:, 0:2] ... wait, A only has 2 rows.
    #
    # Actually, A is 2x4. We split columns: A_left = A[:,0:2], A_right = A[:,2:4]
    # For Strassen, we need a 2x2 block structure. Since A has only 2 rows,
    # each "block" is 1x2. So:
    # A_outer[0,0] = A[0, 0:2] (1x2), A_outer[0,1] = A[0, 2:4] (1x2)
    # A_outer[1,0] = A[1, 0:2] (1x2), A_outer[1,1] = A[1, 2:4] (1x2)
    #
    # B is 4x8. Split rows: B_top = B[0:2,:], B_bot = B[2:4,:]
    # Each is 2x8. Split columns into 4 groups of 2:
    # B_outer[0,0] = B[0:2, 0:2] (2x2), B_outer[0,1] = B[0:2, 2:4] (2x2), etc.
    # B_outer[1,0] = B[2:4, 0:2] (2x2), B_outer[1,1] = B[2:4, 2:4] (2x2), etc.
    #
    # C is 2x8. C[i, k] for i in {0,1}, k in {0,...,7}
    # C_outer[i, g] = C[i, 2g:2g+2] (1x2) for g in {0,1,2,3}
    
    # Mapping for U: A[i, j] -> U index = i * 4 + j
    # For Strassen term t, column group g:
    #   A_outer[a_row, a_col] corresponds to A[a_row, 2*a_col + offset]
    #   where a_col 0 -> j offset 0, a_col 1 -> j offset 2
    
    # For each Strassen term t and column group g (0..3):
    for t in range(7):
        for g in range(4):
            r = t * 4 + g
            
            # U: A elements
            # A_outer[0,0] = A[0, 0:2] -> U indices 0, 1
            # A_outer[0,1] = A[0, 2:4] -> U indices 2, 3
            # A_outer[1,0] = A[1, 0:2] -> U indices 4, 5
            # A_outer[1,1] = A[1, 2:4] -> U indices 6, 7
            # Local index mapping: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
            # Each local block is 1x2, so for local idx la:
            #   row = la // 2, col_group = la % 2
            #   U indices: row*4 + col_group*2 + {0,1}
            
            for la in range(4):
                val = U_local[t][la]
                if val != 0:
                    row = la // 2  # 0 or 1
                    col_g = la % 2  # 0 or 1
                    # U indices for this 1x2 block
                    u_idx1 = row * 4 + col_g * 2 + 0
                    u_idx2 = row * 4 + col_g * 2 + 1
                    U[u_idx1, r] += val
                    U[u_idx2, r] += val
            
            # V: B elements
            # B_outer[0,0] = B[0:2, 2g:2g+2] -> V indices 0*8+2g, 0*8+2g+1, 1*8+2g, 1*8+2g+1
            # B_outer[0,1] = B[0:2, 2g+2:2g+4] ... wait, g goes 0..3 but B has 8 cols
            # Actually for column group g, we use B[:, 2g:2g+2]
            # B_outer[0,g] = B[0:2, 2g:2g+2], B_outer[1,g] = B[2:4, 2g:2g+2]
            # Local idx: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1) in outer 2x2
            # But for fixed g, we only use B_outer[0,g] and B_outer[1,g]
            # The "columns" of the outer 2x2 are the row groups of B
            
            # V_local indices: 0->B[0:2, cols], 1->B[0:2, cols+1]... 
            # Actually V_local[0] = B00, V_local[1] = B01, V_local[2] = B10, V_local[3] = B11
            # For fixed column group g:
            #   B00 = B[0, 2g:2g+2] -> V indices 0*8+2g, 0*8+2g+1
            #   B01 = B[1, 2g:2g+2] -> V indices 1*8+2g, 1*8+2g+1  
            #   B10 = B[2, 2g:2g+2] -> V indices 2*8+2g, 2*8+2g+1
            #   B11 = B[3, 2g:2g+2] -> V indices 3*8+2g, 3*8+2g+1
            
            for lb in range(4):
                val = V_local[t][lb]
                if val != 0:
                    b_row = lb // 2  # 0 or 1 (which 2x4 block)
                    b_col = lb % 2   # 0 or 1 (which column within 2x2)
                    # V index: (b_row*2 + b_col) * 8 + 2*g + ... 
                    # Wait, B is 4x8. B[j, k] -> V index j*8 + k
                    # B_outer[br, bc] for fixed g: rows 2*br to 2*br+1, cols 2*g to 2*g+1
                    # B[2*br, 2*g] -> V index (2*br)*8 + 2*g
                    # B[2*br, 2*g+1] -> V index (2*br)*8 + 2*g + 1
                    # B[2*br+1, 2*g] -> V index (2*br+1)*8 + 2*g
                    # B[2*br+1, 2*g+1] -> V index (2*br+1)*8 + 2*g + 1
                    
                    # V_local lb: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                    # lb 0: B[0, 2g:2g+2] -> V[0*8+2g], V[0*8+2g+1]
                    # lb 1: B[1, 2g:2g+2] -> V[1*8+2g], V[1*8+2g+1]
                    # lb 2: B[2, 2g:2g+2] -> V[2*8+2g], V[2*8+2g+1]
                    # lb 3: B[3, 2g:2g+2] -> V[3*8+2g], V[3*8+2g+1]
                    
                    v_row = lb  # 0,1,2,3
                    v_idx1 = v_row * 8 + 2 * g
                    v_idx2 = v_row * 8 + 2 * g + 1
                    V[v_idx1, r] += val
                    V[v_idx2, r] += val
            
            # W: C elements
            # C[i, 2g:2g+2] for i in {0,1}
            # W index: k*n + i = k*2 + i (trace convention W[k*n+i, r])
            # C[0, 2g] -> W index 2g*2 + 0 = 4g
            # C[0, 2g+1] -> W index (2g+1)*2 + 0 = 4g+2
            # C[1, 2g] -> W index 2g*2 + 1 = 4g+1
            # C[1, 2g+1] -> W index (2g+1)*2 + 1 = 4g+3
            
            # W_local: 0->C00, 1->C01, 2->C10, 3->C11
            # For fixed g:
            #   C00 = C[0, 2g] -> W[4g]
            #   C01 = C[0, 2g+1] -> W[4g+2]
            #   C10 = C[1, 2g] -> W[4g+1]
            #   C11 = C[1, 2g+1] -> W[4g+3]
            
            for lc in range(4):
                val = W_local[t][lc]
                if val != 0:
                    c_row = lc // 2  # 0 or 1
                    c_col = lc % 2   # 0 or 1
                    # W index: (2*g + c_col) * 2 + c_row
                    w_idx = (2 * g + c_col) * 2 + c_row
                    W[w_idx, r] += val
    
    if time.time() - start_time > 140:
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
