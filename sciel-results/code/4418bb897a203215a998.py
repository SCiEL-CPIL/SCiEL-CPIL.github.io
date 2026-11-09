import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition
    # Structure:
    # Split A (3x4) into AL (3x2) and AR (3x2)
    # Split B (4x4) into BT (2x4) and BB (2x4)
    # C = AL*BT + AR*BB
    # Each product is 3x2x4.
    # Split 3x2x4 into two 3x2x2 blocks.
    # Each 3x2x2 block is split into 1x2x2 (rank 4) and 2x2x2 (rank 7).
    # Total rank = 4 * (4 + 7) = 44.
    
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, val=1):
        nonlocal col
        if col >= rank:
            return
        for idx in u_indices:
            U[idx, col] += val
        for idx in v_indices:
            V[idx, col] += val
        for idx in w_indices:
            W[idx, col] += val
        col += 1

    # Define index mapping
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # Process 4 blocks:
    # Block 0: AL * BT1 (A[0:3, 0:2] * B[0:2, 0:2]) -> C[0:3, 0:2]
    # Block 1: AL * BT2 (A[0:3, 0:2] * B[0:2, 2:4]) -> C[0:3, 2:4]
    # Block 2: AR * BB1 (A[0:3, 2:4] * B[2:4, 0:2]) -> C[0:3, 0:2]
    # Block 3: AR * BB2 (A[0:3, 2:4] * B[2:4, 2:4]) -> C[0:3, 2:4]
    
    # Note: The split of A is by columns (j), split of B is by rows (j) and cols (k).
    # AL corresponds to j in {0,1}. AR corresponds to j in {2,3}.
    # BT corresponds to j in {0,1}. BB corresponds to j in {2,3}.
    # BT1 corresponds to k in {0,1}. BT2 corresponds to k in {2,3}.
    # BB1 corresponds to k in {0,1}. BB2 corresponds to k in {2,3}.
    
    for block in range(4):
        # Determine A column offset (j_offset) and B row offset (j_offset)
        # Block 0,1: AL (j=0,1). Block 2,3: AR (j=2,3).
        j_off = 0 if block < 2 else 2
        
        # Determine B col offset (k_offset)
        # Block 0,2: k=0,1. Block 1,3: k=2,3.
        k_off = 0 if block % 2 == 0 else 2
        
        # Sub-block 1: 1x2x2 part (i=0)
        # A[0, j_off:j_off+2] * B[j_off:j_off+2, k_off:k_off+2] -> C[0, k_off:k_off+2]
        # Rank 4 terms
        for jj in range(2):
            for kk in range(2):
                u_idx = 0 * m + (j_off + jj)
                v_idx = (j_off + jj) * p + (k_off + kk)
                w_idx = (k_off + kk) * n + 0
                add_term([u_idx], [v_idx], [w_idx])
        
        # Sub-block 2: 2x2x2 part (i=1,2)
        # A[1:3, j_off:j_off+2] * B[j_off:j_off+2, k_off:k_off+2] -> C[1:3, k_off:k_off+2]
        # Strassen for 2x2x2
        # Local indices:
        # A_loc[0,0] = A[1, j_off], A_loc[0,1] = A[1, j_off+1]
        # A_loc[1,0] = A[2, j_off], A_loc[1,1] = A[2, j_off+1]
        # B_loc[0,0] = B[j_off, k_off], B_loc[0,1] = B[j_off, k_off+1]
        # B_loc[1,0] = B[j_off+1, k_off], B_loc[1,1] = B[j_off+1, k_off+1]
        # C_loc[0,0] = C[1, k_off], C_loc[0,1] = C[1, k_off+1]
        # C_loc[1,0] = C[2, k_off], C_loc[1,1] = C[2, k_off+1]
        
        # Strassen coefficients
        # M1 = (a00+a11)(b00+b11)
        # M2 = a11(b01-b00)
        # M3 = (a00+a01)b11
        # M4 = a00(b10-b00)
        # M5 = (a10+a11)b00
        # M6 = (a00-a10)(b00+b10)
        # M7 = (a01-a11)(b01+b11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M2 + M3 - M7 + M6
        # C10 = M1 + M2 - M4 + M5
        # C11 = M1 - M3 + M4 + M6
        
        # We will add 7 terms.
        # Each term has U, V, W vectors.
        
        # Term 1: M1
        u_idx = [1 * m + j_off, 2 * m + (j_off + 1)]
        v_idx = [(j_off) * p + k_off, (j_off + 1) * p + (k_off + 1)]
        w_idx = [(k_off) * n + 1, (k_off) * n + 2, (k_off + 1) * n + 2]
        # W contributions: C10 (+1), C20 (+1), C21 (+1)
        # Wait, C_loc[1,0] is C[2, k_off]. C_loc[2,0] is C[3, k_off]? No, i=1,2.
        # C_loc[0,0] -> C[1, k_off]. W[0, 1]
        # C_loc[0,1] -> C[1, k_off+1]. W[1, 1]
        # C_loc[1,0] -> C[2, k_off]. W[0, 2]
        # C_loc[1,1] -> C[2, k_off+1]. W[1, 2]
        
        # M1 coeffs in C:
        # C00: +1 -> W[k_off, 1]
        # C01: 0
        # C10: +1 -> W[k_off, 2]
        # C11: +1 -> W[k_off+1, 2]
        w_idx = [(k_off) * n + 1, (k_off) * n + 2, (k_off + 1) * n + 2]
        add_term(u_idx, v_idx, w_idx)
        
        # Term 2: M2
        u_idx = [2 * m + (j_off + 1)] # a11
        v_idx = [(j_off) * p + (k_off + 1), (j_off) * p + k_off] # b01 - b00
        # Coeffs: C01 +1, C10 +1
        # C01 -> W[k_off+1, 1]
        # C10 -> W[k_off, 2]
        w_idx = [(k_off + 1) * n + 1, (k_off) * n + 2]
        add_term(u_idx, v_idx, w_idx)
        
        # Term 3: M3
        u_idx = [1 * m + j_off, 1 * m + (j_off + 1)] # a00 + a01
        v_idx = [(j_off + 1) * p + (k_off + 1)] # b11
        # Coeffs: C01 +1, C11 -1
        # C01 -> W[k_off+1, 1]
        # C11 -> W[k_off+1, 2]
        w_idx = [(k_off + 1) * n + 1, (k_off + 1) * n + 2]
        # Need to handle signs. add_term only adds.
        # We can pass negative values or handle in W.
        # Let's modify add_term to accept a list of (idx, val) or just assume we can set W directly.
        # Easier: Just set W values directly in a helper.
        
        # Let's refactor to a more direct construction to handle signs easily.
        
    # Reset and use direct assignment for clarity and correctness with signs
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    
    def add_term_signed(u_list, v_list, w_list):
        nonlocal col
        for idx, val in u_list:
            U[idx, col] += val
        for idx, val in v_list:
            V[idx, col] += val
        for idx, val in w_list:
            W[idx, col] += val
        col += 1

    for block in range(4):
        j_off = 0 if block < 2 else 2
        k_off = 0 if block % 2 == 0 else 2
        
        # 1x2x2 part
        for jj in range(2):
            for kk in range(2):
                u = [(0 * m + j_off + jj, 1)]
                v = [((j_off + jj) * p + k_off + kk, 1)]
                w = [((k_off + kk) * n + 0, 1)]
                add_term_signed(u, v, w)
        
        # 2x2x2 part
        # M1
        u = [(1 * m + j_off, 1), (2 * m + j_off + 1, 1)]
        v = [(j_off * p + k_off, 1), ((j_off + 1) * p + k_off + 1, 1)]
        w = [((k_off) * n + 1, 1), ((k_off) * n + 2, 1), ((k_off + 1) * n + 2, 1)]
        add_term_signed(u, v, w)
        
        # M2
        u = [(2 * m + j_off + 1, 1)]
        v = [(j_off * p + k_off + 1, 1), (j_off * p + k_off, -1)]
        w = [((k_off + 1) * n + 1, 1), ((k_off) * n + 2, 1)]
        add_term_signed(u, v, w)
        
        # M3
        u = [(1 * m + j_off, 1), (1 * m + j_off + 1, 1)]
        v = [((j_off + 1) * p + k_off + 1, 1)]
        w = [((k_off + 1) * n + 1, 1), ((k_off + 1) * n + 2, -1)]
        add_term_signed(u, v, w)
        
        # M4
        u = [(1 * m + j_off, 1)]
        v = [((j_off + 1) * p + k_off, 1), (j_off * p + k_off, -1)]
        w = [((k_off) * n + 1, 1), ((k_off + 1) * n + 2, 1)]
        add_term_signed(u, v, w)
        
        # M5
        u = [(2 * m + j_off, 1), (2 * m + j_off + 1, 1)]
        v = [(j_off * p + k_off, 1)]
        w = [((k_off) * n + 1, -1), ((k_off) * n + 2, 1)]
        add_term_signed(u, v, w)
        
        # M6
        u = [(1 * m + j_off, 1), (2 * m + j_off, -1)]
        v = [(j_off * p + k_off, 1), ((j_off + 1) * p + k_off, 1)]
        w = [((k_off + 1) * n + 1, 1), ((k_off + 1) * n + 2, 1)]
        add_term_signed(u, v, w)
        
        # M7
        u = [(1 * m + j_off + 1, 1), (2 * m + j_off + 1, -1)]
        v = [(j_off * p + k_off + 1, 1), ((j_off + 1) * p + k_off + 1, 1)]
        w = [((k_off) * n + 1, 1), ((k_off + 1) * n + 1, -1)]
        add_term_signed(u, v, w)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
