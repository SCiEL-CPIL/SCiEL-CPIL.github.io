import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 + Rank 44 for 4x4x3 = 93
    rank = 93
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # --- Part 1: 4x4x4 (Rank 49) ---
    # Strassen decomposition for M2 (Rank 7)
    # Local indices for 2x2 matrices: 0,1,2,3
    # A: 00, 01, 10, 11
    # B: 00, 01, 10, 11
    # C: 00, 01, 10, 11
    
    # Strassen terms for M2
    # U_local, V_local, W_local
    strassen_m2 = [
        # 1: (A00+A11)(B00+B11)
        (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
        # 2: (A10+A11)B00
        (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, -1])),
        # 3: A00(B01-B11)
        (np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]), np.array([0, 1, 0, 1])),
        # 4: A11(B10-B00)
        (np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0]), np.array([1, 0, 1, 0])),
        # 5: (A00+A01)B11
        (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([-1, 1, 0, 0])),
        # 6: (A10-A00)(B00+B01)
        (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1])),
        # 7: (A01-A11)(B10+B11)
        (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, 0, 1]))
    ]
    
    # Block mappings for 4x4 matrices
    # A blocks (U indices):
    # A11: 0,1,4,5
    # A12: 2,3,6,7
    # A21: 8,9,12,13
    # A22: 10,11,14,15
    a_blocks = [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15]
    ]
    
    # B blocks (V indices) for 4x4 (p=4):
    # B11: 0,1,4,5
    # B12: 2,3,6,7
    # B21: 8,9,12,13
    # B22: 10,11,14,15
    b_blocks_4 = [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15]
    ]
    
    # C blocks (W indices) for 4x4 (p=4, n=4):
    # C11: 0,1,4,5
    # C12: 2,3,6,7
    # C21: 8,9,12,13
    # C22: 10,11,14,15
    c_blocks_4 = [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15]
    ]
    
    # Strassen coefficients for M4 output blocks
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    # P1..P7 are the 7 terms.
    # We iterate through the 7 terms.
    # For each term k (0-indexed), we determine which output blocks it contributes to and with what sign.
    
    # Term 0 (P1): C11 (+1), C22 (+1)
    # Term 1 (P2): C21 (+1), C22 (-1)
    # Term 2 (P3): C12 (+1), C22 (+1)
    # Term 3 (P4): C11 (+1), C21 (+1)
    # Term 4 (P5): C11 (-1), C12 (+1)
    # Term 5 (P6): C22 (+1)
    # Term 6 (P7): C11 (+1), C22 (+1)
    
    contrib = [
        [(0, 1), (3, 1)],   # P1
        [(2, 1), (3, -1)],  # P2
        [(1, 1), (3, 1)],   # P3
        [(0, 1), (2, 1)],   # P4
        [(0, -1), (1, 1)],  # P5
        [(3, 1)],           # P6
        [(0, 1), (3, 1)]    # P7
    ]
    
    # Input block combinations for each term
    # P1: (A11+A22)(B11+B22) -> A: 0,3; B: 0,3
    # P2: (A21+A22)B11       -> A: 2,3; B: 0
    # P3: A11(B12-B22)       -> A: 0; B: 1,3
    # P4: A22(B21-B11)       -> A: 3; B: 2,0
    # P5: (A11+A12)B22       -> A: 0,1; B: 3
    # P6: (A21-A11)(B11+B12) -> A: 2,0; B: 0,1
    # P7: (A12-A22)(B21+B22) -> A: 1,3; B: 2,3
    
    in_a = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)]
    ]
    
    in_b = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)]
    ]
    
    col = 0
    for k in range(7):
        u_loc, v_loc, w_loc = strassen_m2[k]
        
        # Construct global U
        u_global = np.zeros(16)
        for block_idx, sign in in_a[k]:
            indices = a_blocks[block_idx]
            u_global[indices] += sign * u_loc
            
        # Construct global V
        v_global = np.zeros(16) # Note: V size is m*p. For 4x4x4, m=4, p=4, size 16.
        for block_idx, sign in in_b[k]:
            indices = b_blocks_4[block_idx]
            v_global[indices] += sign * v_loc
            
        # Construct global W
        w_global = np.zeros(16)
        for block_idx, sign in contrib[k]:
            indices = c_blocks_4[block_idx]
            w_global[indices] += sign * w_loc
            
        U[:, col] = u_global
        V[:, col] = v_global
        W[:, col] = w_global
        col += 1
        
    # --- Part 2: 4x4x3 (Rank 44) ---
    # Split B into B1 (4x2) and B2 (4x1)
    # B1: 4 multiplications of M2x2x2 (Rank 7 each) -> 28
    # B2: 4 multiplications of M2x2x1 (Rank 4 each) -> 16
    
    # B1 blocks (V indices for p=3):
    # B11: j in {0,1}, k in {0,1}. Indices: 0,1,3,4
    # B21: j in {2,3}, k in {0,1}. Indices: 6,7,9,10
    b_blocks_3_1 = [
        [0, 1, 3, 4],
        [6, 7, 9, 10]
    ]
    
    # B2 blocks (V indices for p=3):
    # B12: j in {0,1}, k in {2}. Indices: 2,5
    # B22: j in {2,3}, k in {2}. Indices: 8,11
    b_blocks_3_2 = [
        [2, 5],
        [8, 11]
    ]
    
    # C blocks (W indices for p=3, n=4):
    # C11: i in {0,1}, k in {0,1}. Indices: 0,1,4,5
    # C12: i in {0,1}, k in {2}. Indices: 2,3
    # C21: i in {2,3}, k in {0,1}. Indices: 8,9,12,13
    # C22: i in {2,3}, k in {2}. Indices: 10,11
    c_blocks_3 = [
        [0, 1, 4, 5],
        [2, 3],
        [8, 9, 12, 13],
        [10, 11]
    ]
    
    # For B1 (4x2), we use 4 terms of M2x2x2.
    # Term 0: A11 * B11
    # Term 1: A12 * B21
    # Term 2: A21 * B11
    # Term 3: A22 * B21
    
    # M2x2x2 decomposition (Rank 7)
    # We can reuse strassen_m2.
    # For each of the 4 block multiplications, we use the 7 Strassen terms.
    
    # Mapping for B1 part
    # A blocks: 0,1,2,3
    # B blocks: 0 (B11), 1 (B21)
    # C blocks: 0 (C11), 1 (C12) for top row? No.
    # C1 = [A11 B11 + A12 B21; A21 B11 + A22 B21]
    # C11 (top-left 2x2 of C1) = A11 B11
    # C12 (top-right 2x2 of C1) = A12 B21
    # C21 (bottom-left 2x2 of C1) = A21 B11
    # C22 (bottom-right 2x2 of C1) = A22 B21
    
    # So for A11*B11: A_block=0, B_block=0, C_block=0
    # For A12*B21: A_block=1, B_block=1, C_block=1
    # For A21*B11: A_block=2, B_block=0, C_block=2
    # For A22*B21: A_block=3, B_block=1, C_block=3
    
    b1_map = [
        (0, 0, 0),
        (1, 1, 1),
        (2, 0, 2),
        (3, 1, 3)
    ]
    
    for a_blk, b_blk, c_blk in b1_map:
        for k in range(7):
            u_loc, v_loc, w_loc = strassen_m2[k]
            
            u_global = np.zeros(16)
            indices_a = a_blocks[a_blk]
            u_global[indices_a] = u_loc
            
            v_global = np.zeros(28) # m*p = 4*3 = 12? No, m=4, p=7. V size is 28.
            # Wait, V size is m*p = 4*7 = 28.
            # In Part 1, we used V size 16 (for p=4).
            # We must ensure V is size 28.
            # In Part 1, we only filled indices 0..15. Indices 16..27 are 0.
            # This is correct because p=7, so k goes up to 6.
            # B11 in 4x4x4 used k=0,1.
            # B11 in 4x4x3 uses k=0,1.
            
            indices_b = b_blocks_3_1[b_blk]
            v_global[indices_b] = v_loc
            
            w_global = np.zeros(28) # p*n = 7*4 = 28.
            indices_c = c_blocks_3[c_blk]
            w_global[indices_c] = w_loc
            
            U[:, col] = u_global
            V[:, col] = v_global
            W[:, col] = w_global
            col += 1
            
    # For B2 (4x1), we use 4 terms of M2x2x1.
    # M2x2x1 (Rank 4)
    # A is 2x2, B is 2x1, C is 2x1.
    # Schoolbook:
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    # Rank 4 terms:
    # 1: A00 B0 -> C0
    # 2: A01 B1 -> C0
    # 3: A10 B0 -> C1
    # 4: A11 B1 -> C1
    
    strassen_m21 = [
        (np.array([1, 0, 0, 0]), np.array([1, 0]), np.array([1, 0])),
        (np.array([0, 1, 0, 0]), np.array([0, 1]), np.array([1, 0])),
        (np.array([0, 0, 1, 0]), np.array([1, 0]), np.array([0, 1])),
        (np.array([0, 0, 0, 1]), np.array([0, 1]), np.array([0, 1]))
    ]
    
    # Mapping for B2 part
    # C2 = [A11 B12 + A12 B22; A21 B12 + A22 B22]
    # C12 (top-right 2x1 of C2) = A11 B12
    # C22 (bottom-right 2x1 of C2) = A21 B12? No.
    # C2 is 4x1.
    # Top 2 rows: A11 B12 + A12 B22
    # Bottom 2 rows: A21 B12 + A22 B22
    
    # So:
    # A11 * B12 -> C top (indices 2,3 in W? No, C12 is indices 2,3)
    # A12 * B22 -> C top
    # A21 * B12 -> C bottom (indices 10,11)
    # A22 * B22 -> C bottom
    
    b2_map = [
        (0, 0, 1), # A11, B12, C12
        (1, 1, 1), # A12, B22, C12
        (2, 0, 3), # A21, B12, C22
        (3, 1, 3)  # A22, B22, C22
    ]
    
    for a_blk, b_blk, c_blk in b2_map:
        for k in range(4):
            u_loc, v_loc, w_loc = strassen_m21[k]
            
            u_global = np.zeros(16)
            indices_a = a_blocks[a_blk]
            u_global[indices_a] = u_loc
            
            v_global = np.zeros(28)
            indices_b = b_blocks_3_2[b_blk]
            v_global[indices_b] = v_loc
            
            w_global = np.zeros(28)
            indices_c = c_blocks_3[c_blk]
            w_global[indices_c] = w_loc
            
            U[:, col] = u_global
            V[:, col] = v_global
            W[:, col] = w_global
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

