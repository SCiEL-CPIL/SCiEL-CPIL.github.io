import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Total rank estimate: 21 (A1 B1) + 21 (A2a B2a) + 1 (A2b B2b) = 43
    total_rank = 43
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Define Strassen 2x2x2 factors
    # U_s[i*2+j, r], V_s[j*2+k, r], W_s[k*2+i, r]
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # M0 = (A00+A11)(B00+B11)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[1, 0] = -1; W_s[2, 0] = -1; W_s[3, 0] = 1
    
    # M1 = (A10+A11)B00
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1; W_s[3, 1] = -1
    
    # M2 = A00(B10-B11)
    U_s[0, 2] = 1
    V_s[2, 2] = 1; V_s[3, 2] = -1
    W_s[1, 2] = 1; W_s[3, 2] = 1
    
    # M3 = A11(B01-B10)
    U_s[3, 3] = 1
    V_s[1, 3] = 1; V_s[2, 3] = -1
    W_s[0, 3] = 1
    
    # M4 = (A00+A10)B11
    U_s[0, 4] = 1; U_s[2, 4] = 1
    V_s[3, 4] = 1
    W_s[2, 4] = -1; W_s[3, 4] = 1
    
    # M5 = (A01+A11)(B00+B10)
    U_s[1, 5] = 1; U_s[3, 5] = 1
    V_s[0, 5] = 1; V_s[2, 5] = 1
    W_s[0, 5] = -1; W_s[1, 5] = 1
    
    # M6 = (A01-A10)(B11+B01)
    U_s[1, 6] = 1; U_s[2, 6] = -1
    V_s[3, 6] = 1; V_s[1, 6] = 1
    W_s[0, 6] = 1; W_s[2, 6] = 1
    
    rank_offset = 0
    
    # Helper to add a Strassen block
    def add_strassen_block(a_i_start, a_j_start, b_j_start, b_k_start, w_i_start, w_k_start):
        nonlocal rank_offset
        # a_i_start: 0 (since n=2, only 1 block of rows)
        # a_j_start: start index in A's columns (j)
        # b_j_start: start index in B's rows (j)
        # b_k_start: start index in B's columns (k)
        # w_i_start: 0
        # w_k_start: start index in C's columns (k)
        
        for r in range(7):
            # U: A indices
            for i in range(2):
                for j_l in range(2):
                    U[i * m + (a_j_start + j_l), rank_offset] = U_s[i * 2 + j_l, r]
            
            # V: B indices
            for j_l in range(2):
                for k_l in range(2):
                    V[(b_j_start + j_l) * p + (b_k_start + k_l), rank_offset] = V_s[j_l * 2 + k_l, r]
            
            # W: C indices (p*n, index k*n + i)
            for i in range(2):
                for k_l in range(2):
                    W[(b_k_start + k_l) * n + i, rank_offset] = W_s[k_l * 2 + i, r]
            
            rank_offset += 1
            
    # Helper to add a Rank-1 block (2x1x6)
    def add_rank1_block(a_i_start, a_j_idx, b_j_idx, b_k_start, w_i_start, w_k_start):
        nonlocal rank_offset
        # A: 2x1 at (i, a_j_idx)
        # B: 1x6 at (b_j_idx, k)
        # C: 2x6 at (i, k)
        
        for i in range(2):
            U[i * m + a_j_idx, rank_offset] = 1
        for k in range(6):
            V[b_j_idx * p + (b_k_start + k), rank_offset] = 1
        for i in range(2):
            for k in range(6):
                W[(b_k_start + k) * n + i, rank_offset] = 1
        
        rank_offset += 1

    # Part 1: A1 (2x2, j=0,1) * B1 (2x6, j=0,1)
    # Split B1 into 3 blocks of 2x2
    add_strassen_block(0, 0, 0, 0, 0, 0) # B cols 0,1
    add_strassen_block(0, 0, 0, 2, 0, 2) # B cols 2,3
    add_strassen_block(0, 0, 0, 4, 0, 4) # B cols 4,5
    
    # Part 2: A2 (2x3, j=2,3,4) * B2 (3x6, j=2,3,4)
    # Split A2 into A2a (2x2, j=2,3) and A2b (2x1, j=4)
    # Split B2 into B2a (2x6, j=2,3) and B2b (1x6, j=4)
    
    # A2a * B2a (2x2 * 2x6)
    add_strassen_block(0, 2, 2, 0, 0, 0) # B cols 0,1
    add_strassen_block(0, 2, 2, 2, 0, 2) # B cols 2,3
    add_strassen_block(0, 2, 2, 4, 0, 4) # B cols 4,5
    
    # A2b * B2b (2x1 * 1x6)
    add_rank1_block(0, 4, 4, 0, 0, 0)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
