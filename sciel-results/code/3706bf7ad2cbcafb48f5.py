"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2_2x2():
    """
    Returns rank-7 decomposition for 2x2 * 2x2.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    n, m, p = 2, 2, 2
    r = 7
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    
    # Strassen's algorithm for 2x2 matrices
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[a+e, b+f], [c+g, d+h]]? No, C = AB.
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices for A (2x2): 0:00, 1:01, 2:10, 3:11
    # Indices for B (2x2): 0:00, 1:01, 2:10, 3:11
    # Indices for C (2x2): 0:00, 1:01, 2:10, 3:11
    
    # M1: (A00+A11) * (B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A10+A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1 # Wait, C11 = M1 - M2 + ...
    # Let's check the formula again.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1 contributes +1 to C00 and C11
    W[0, 0] = 1
    W[3, 0] = 1
    
    # M2: (A10+A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # M2 contributes +1 to C10 and -1 to C11
    W[2, 1] = 1
    W[3, 1] = -1
    
    # M3: A00 * (B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # M3 contributes +1 to C01 and C11
    W[1, 2] = 1
    W[3, 2] = 1
    
    # M4: A11 * (B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # M4 contributes +1 to C00 and C10
    W[0, 3] = 1
    W[2, 3] = 1
    
    # M5: (A00+A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # M5 contributes -1 to C00 and +1 to C01
    W[0, 4] = -1
    W[1, 4] = 1
    
    # M6: (A10-A00) * (B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # M6 contributes +1 to C11
    W[3, 5] = 1
    
    # M7: (A01-A11) * (B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # M7 contributes +1 to C00
    W[0, 6] = 1
    
    return U, V, W

def _construct_2x3_3x2_rank11():
    """
    Rank-11 decomposition for 2x3 * 3x2.
    Splits A (2x3) into A_L (2x2) and A_R (2x1).
    Splits B (3x2) into B_T (2x2) and B_B (1x2).
    C = A_L B_T + A_R B_B.
    A_L B_T is 2x2*2x2 (rank 7).
    A_R B_B is 2x1*1x2 (rank 4).
    Total rank 11.
    """
    n, m, p = 2, 3, 2
    r_strassen = 7
    r_outer = 4
    r_total = r_strassen + r_outer
    
    U = np.zeros((n*m, r_total))
    V = np.zeros((m*p, r_total))
    W = np.zeros((p*n, r_total))
    
    # 1. Strassen part for A_L (cols 0,1 of A) and B_T (rows 0,1 of B)
    U_s, V_s, W_s = _strassen_2x2_2x2()
    
    # Map Strassen indices to full problem indices
    # A_L indices in full A (2x3):
    # A_L[0,0] -> A[0,0] (idx 0)
    # A_L[0,1] -> A[0,1] (idx 1)
    # A_L[1,0] -> A[1,0] (idx 3)
    # A_L[1,1] -> A[1,1] (idx 4)
    # Strassen U indices: 0:00, 1:01, 2:10, 3:11
    # Map: 0->0, 1->1, 2->3, 3->4
    
    # B_T indices in full B (3x2):
    # B_T[0,0] -> B[0,0] (idx 0)
    # B_T[0,1] -> B[0,1] (idx 1)
    # B_T[1,0] -> B[1,0] (idx 2)
    # B_T[1,1] -> B[1,1] (idx 3)
    # Strassen V indices: 0:00, 1:01, 2:10, 3:11
    # Map: 0->0, 1->1, 2->2, 3->3
    
    # C indices are same for 2x2.
    
    for c in range(r_strassen):
        for i in range(4):
            # U_s[i, c] is coefficient for A_L[i//2, i%2]
            # We need to map i to full A index
            if i == 0: full_i = 0
            elif i == 1: full_i = 1
            elif i == 2: full_i = 3
            elif i == 3: full_i = 4
            else: full_i = 0 # shouldn't happen
            
            U[full_i, c] = U_s[i, c]
            
        for j in range(4):
            # V_s[j, c] is coefficient for B_T[j//2, j%2]
            # Map j to full B index
            if j == 0: full_j = 0
            elif j == 1: full_j = 1
            elif j == 2: full_j = 2
            elif j == 3: full_j = 3
            else: full_j = 0
            
            V[full_j, c] = V_s[j, c]
            
        for k in range(4):
            # W_s[k, c] is coefficient for C[k//2, k%2]
            # C indices are same
            W[k, c] = W_s[k, c]
            
    # 2. Outer product part for A_R (col 2 of A) and B_B (row 2 of B)
    # A_R is 2x1: A[0,2] (idx 2), A[1,2] (idx 5)
    # B_B is 1x2: B[2,0] (idx 4), B[2,1] (idx 5)
    # Products:
    # P1: A[0,2] * B[2,0] -> C[0,0]
    # P2: A[0,2] * B[2,1] -> C[0,1]
    # P3: A[1,2] * B[2,0] -> C[1,0]
    # P4: A[1,2] * B[2,1] -> C[1,1]
    
    offset = r_strassen
    # P1
    U[2, offset] = 1
    V[4, offset] = 1
    W[0, offset] = 1
    
    # P2
    U[2, offset+1] = 1
    V[5, offset+1] = 1
    W[1, offset+1] = 1
    
    # P3
    U[5, offset+2] = 1
    V[4, offset+2] = 1
    W[2, offset+2] = 1
    
    # P4
    U[5, offset+3] = 1
    V[5, offset+3] = 1
    W[3, offset+3] = 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Split B (3x8) into 4 blocks of 3x2
    # B1: cols 0-1
    # B2: cols 2-3
    # B3: cols 4-5
    # B4: cols 6-7
    
    # Each block uses rank-11 decomposition
    U_block, V_block, W_block = _construct_2x3_3x2_rank11()
    r_block = len(U_block[0])
    
    total_rank = 4 * r_block
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    for block_idx in range(4):
        offset = block_idx * r_block
        
        # U is the same for all blocks (since A is the same)
        U[:, offset:offset+r_block] = U_block
        
        # V is different for each block (different columns of B)
        # B_block is B[:, block_idx*2 : block_idx*2+2]
        # V_block indices are 0..5 corresponding to B rows 0,1,2 and cols 0,1
        # We need to map these to full B indices
        # Full B index = row * 8 + col
        # V_block[j*2 + k] corresponds to B[j, k] in the block
        # So full index = j * 8 + (block_idx*2 + k)
        
        for c in range(r_block):
            for j in range(3):
                for k in range(2):
                    v_idx = j*2 + k
                    full_v_idx = j*8 + (block_idx*2 + k)
                    V[full_v_idx, offset+c] = V_block[v_idx, c]
                    
        # W is different for each block (different columns of C)
        # C_block is C[:, block_idx*2 : block_idx*2+2]
        # W_block indices are 0..3 corresponding to C rows 0,1 and cols 0,1
        # Full C index = row * 2 + col? No, W[k*n+i, c].
        # W_block[k*2 + i] corresponds to C[i, k] in the block?
        # Let's check the convention in _construct_2x3_3x2_rank11.
        # W[k*2 + i, c] where k is col, i is row.
        # So W_block index = k*2 + i.
        # Full C index = k_full * 2 + i_full.
        # k_full = block_idx*2 + k_block.
        # i_full = i_block.
        
        for c in range(r_block):
            for i in range(2):
                for k in range(2):
                    w_idx = k*2 + i
                    full_w_idx = (block_idx*2 + k)*2 + i
                    W[full_w_idx, offset+c] = W_block[w_idx, c]
                    
    # Convert to lists for output
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
