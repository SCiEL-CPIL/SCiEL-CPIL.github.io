import numpy as np
import time

DIMENSIONS = [3, 4, 5]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140  # 10s safety margin before 150s limit

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    Uses Strassen's algorithm for 2x2x2 blocks to achieve rank 56.
    The decomposition is given by U, V, W such that T = sum_r U[:, r] outer V[:, r] outer W[:, r].
    Indices:
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank) with W[k*n+i, r]
    """
    n, m, p = DIMENSIONS
    
    if time.time() > _DEADLINE:
        # Return a valid fallback (schoolbook, rank 60) if we're running out of time
        return _schoolbook_fallback(n, m, p)
    
    # Total rank will be 56: 36 from 2x4x5 part + 20 from 1x4x5 part
    # We build the decomposition in parts.
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Part 1: 2x4x5 multiplication (rows 0,1 of A)
    # A1 is 2x4, B is 4x5, C1 is 2x5
    # We can decompose 2x4x5 as two 2x2x5 blocks? No, 2x4x5 = 2x(2+2)x5.
    # Actually, 2x4x5 can be seen as multiplying a 2x4 matrix by a 4x5 matrix.
    # We can split B into two 2x5 blocks? No, B is 4x5.
    # Let's split the middle dimension m=4 into two parts of size 2.
    # A1 (2x4) * B (4x5) = A1[:, 0:2] * B[0:2, :] + A1[:, 2:4] * B[2:4, :]
    # Each term is 2x2x5.
    # 2x2x5 rank? Schoolbook is 20. Strassen on 2x2x2 blocks?
    # 2x2x5 = 2x2x(2+3) = 2x2x2 + 2x2x3.
    # Rank(2x2x2) = 7 (Strassen).
    # Rank(2x2x3) = ? Schoolbook is 12. Can we do better?
    # Actually, let's just use Strassen for the 2x2x2 part and schoolbook for 2x2x3?
    # Or simpler: use Strassen for 2x2x2 and 2x2x2 and 2x2x1?
    # 2x2x5 = 2x2x2 + 2x2x2 + 2x2x1.
    # Rank = 7 + 7 + 2 = 16.
    # So 2x4x5 = 2 * 16 = 32?
    # Wait, A1[:, 0:2] * B[0:2, :] is 2x2x5.
    # A1[:, 2:4] * B[2:4, :] is 2x2x5.
    # Total rank for 2x4x5 part = 16 + 16 = 32.
    # Part 2: 1x4x5 multiplication (row 2 of A)
    # A2 is 1x4, B is 4x5, C2 is 1x5.
    # 1x4x5 = 1x(2+2)x5 = 1x2x5 + 1x2x5.
    # 1x2x5 = 1x2x(2+2+1) = 1x2x2 + 1x2x2 + 1x2x1.
    # Rank(1x2x2) = 2 (schoolbook is 4, but 1x2x2 is just dot products, rank 2).
    # Actually, 1x2x2: A is 1x2, B is 2x2, C is 1x2.
    # C0 = A0*B00 + A1*B10
    # C1 = A0*B01 + A1*B11
    # This is 2 multiplications. Rank 2.
    # Rank(1x2x1) = 1.
    # So Rank(1x2x5) = 2 + 2 + 1 = 5.
    # Total rank for 1x4x5 part = 5 + 5 = 10.
    # Total rank = 32 + 10 = 42.
    # This is much better than 56!
    
    # Let's verify the ranks.
    # 2x2x2 Strassen rank 7.
    # 2x2x1 schoolbook rank 2.
    # 1x2x2 rank 2.
    # 1x2x1 rank 1.
    
    # We need to construct the explicit decompositions.
    
    # Helper to add a block decomposition
    def add_block(u_offset, v_offset, w_offset, u_cols, v_cols, w_cols, U_block, V_block, W_block):
        """
        Adds a block decomposition to the global lists.
        u_offset, v_offset, w_offset are the starting indices in the global U, V, W.
        u_cols, v_cols, w_cols are the sizes of the blocks in U, V, W.
        U_block, V_block, W_block are the factor matrices for the block.
        """
        for r in range(U_block.shape[1]):
            U_parts.append(np.zeros(n * m))
            U_parts[-1][u_offset : u_offset + u_cols] = U_block[:, r]
            
            V_parts.append(np.zeros(m * p))
            V_parts[-1][v_offset : v_offset + v_cols] = V_block[:, r]
            
            W_parts.append(np.zeros(p * n))
            W_parts[-1][w_offset : w_offset + w_cols] = W_block[:, r]
    
    # 1. Strassen 2x2x2 decomposition
    # U: 4x7, V: 4x7, W: 4x7
    # U indices: 0:A00, 1:A01, 2:A10, 3:A11
    # V indices: 0:B00, 1:B01, 2:B10, 3:B11
    # W indices: 0:C00, 1:C01, 2:C10, 3:C11
    def strassen_2x2x2():
        """
        Standard Strassen decomposition for 2x2 matrix multiplication.
        C = A*B where A, B, C are 2x2.
        C00 = A00*B00 + A01*B10
        C01 = A00*B01 + A01*B11
        C10 = A10*B00 + A11*B10
        C11 = A10*B01 + A11*B11
        
        Strassen:
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)(B00)
        M3 = (A00)(B01-B11)
        M4 = (A11)(B10-B00)
        M5 = (A00+A01)(B11)
        M6 = (A10-A00)(B00+B01)
        M7 = (A01-A11)(B10+B11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        # U: A00=1, A11=1
        U[0, 0] = 1; U[3, 0] = 1
        # V: B00=1, B11=1
        V[0, 0] = 1; V[3, 0] = 1
        # W: C00+=1, C11+=1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)(B00)
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        # W: C10+=1, C11-=1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = (A00)(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        # W: C01+=1, C11+=1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = (A11)(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        # W: C00+=1, C10+=1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A00+A01)(B11)
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        # W: C00-=1, C01+=1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        # W: C11+=1
        W[3, 5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        # W: C00+=1
        W[0, 6] = 1
        
        return U, V, W
    
    # 2. Schoolbook 2x2x1 decomposition
    # A (2x2) * B (2x1) = C (2x1)
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    # Rank 4.
    def schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        
        # Term 1: A00*B0 -> C0
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # Term 2: A01*B1 -> C0
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        # Term 3: A10*B0 -> C1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        # Term 4: A11*B1 -> C1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        
        return U, V, W
    
    # 3. Schoolbook 1x2x2 decomposition
    # A (1x2) * B (2x2) = C (1x2)
    # C0 = A0*B00 + A1*B10
    # C1 = A0*B01 + A1*B11
    # Rank 4.
    def schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        
        # Term 1: A0*B00 -> C0
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # Term 2: A1*B10 -> C0
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        # Term 3: A0*B01 -> C1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        # Term 4: A1*B11 -> C1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        
        return U, V, W
    
    # 4. Schoolbook 1x2x1 decomposition
    # A (1x2) * B (2x1) = C (1x1)
    # C0 = A0*B0 + A1*B1
    # Rank 2.
    def schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        
        # Term 1: A0*B0
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # Term 2: A1*B1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        
        return U, V, W

    # Now we assemble the 3x4x5 multiplication.
    # n=3, m=4, p=5.
    # A is 3x4, B is 4x5, C is 3x5.
    # We split A into A1 (2x4) and A2 (1x4).
    # We split B into B1 (2x5) and B2 (2x5) along the middle dimension?
    # No, B is 4x5. We split the rows of B into two groups of 2.
    # B = [B_top; B_bottom] where B_top is 2x5 and B_bottom is 2x5.
    # A1 = [A1_top, A1_bottom] where A1_top is 2x2 and A1_bottom is 2x2.
    # C1 = A1 * B = A1_top * B_top + A1_bottom * B_bottom.
    # Each term is 2x2x5.
    # We split 2x2x5 into 2x2x2 + 2x2x2 + 2x2x1.
    # B_top is 2x5. Split into B_top_left (2x2), B_top_mid (2x2), B_top_right (2x1).
    # So A1_top * B_top = A1_top * B_top_left + A1_top * B_top_mid + A1_top * B_top_right.
    # Each is 2x2x2, 2x2x2, 2x2x1.
    
    # Let's define the indices.
    # U indices: 0..11 (3x4)
    # V indices: 0..19 (4x5)
    # W indices: 0..14 (5x3) -> W[k*n+i, r]
    
    # Part 1: A1 (rows 0,1) * B
    # A1_top (2x2): U indices 0,1 (row 0), 4,5 (row 1). Wait, U is flattened row-major.
    # U[i*m + j].
    # Row 0: j=0,1,2,3 -> indices 0,1,2,3.
    # Row 1: j=0,1,2,3 -> indices 4,5,6,7.
    # Row 2: j=0,1,2,3 -> indices 8,9,10,11.
    
    # A1_top: rows 0,1; cols 0,1. Indices: 0,1,4,5.
    # A1_bottom: rows 0,1; cols 2,3. Indices: 2,3,6,7.
    
    # B_top: rows 0,1; cols 0..4. Indices: 0,1,2,3,4, 5,6,7,8,9.
    # B_bottom: rows 2,3; cols 0..4. Indices: 10,11,12,13,14, 15,16,17,18,19.
    
    # B_top_left: rows 0,1; cols 0,1. Indices: 0,1,5,6.
    # B_top_mid: rows 0,1; cols 2,3. Indices: 2,3,7,8.
    # B_top_right: rows 0,1; col 4. Indices: 4,9.
    
    # B_bottom_left: rows 2,3; cols 0,1. Indices: 10,11,15,16.
    # B_bottom_mid: rows 2,3; cols 2,3. Indices: 12,13,17,18.
    # B_bottom_right: rows 2,3; col 4. Indices: 14,19.
    
    # C1_top: rows 0; cols 0..4. W indices: k*n+i.
    # C1_top, col 0: k=0, i=0 -> 0*3+0 = 0.
    # C1_top, col 1: k=1, i=0 -> 1*3+0 = 3.
    # C1_top, col 2: k=2, i=0 -> 2*3+0 = 6.
    # C1_top, col 3: k=3, i=0 -> 3*3+0 = 9.
    # C1_top, col 4: k=4, i=0 -> 4*3+0 = 12.
    
    # C1_bottom: rows 1; cols 0..4.
    # C1_bottom, col 0: k=0, i=1 -> 0*3+1 = 1.
    # C1_bottom, col 1: k=1, i=1 -> 1*3+1 = 4.
    # C1_bottom, col 2: k=2, i=1 -> 2*3+1 = 7.
    # C1_bottom, col 3: k=3, i=1 -> 3*3+1 = 10.
    # C1_bottom, col 4: k=4, i=1 -> 4*3+1 = 13.
    
    # C2: rows 2; cols 0..4.
    # C2, col 0: k=0, i=2 -> 0*3+2 = 2.
    # C2, col 1: k=1, i=2 -> 1*3+2 = 5.
    # C2, col 2: k=2, i=2 -> 2*3+2 = 8.
    # C2, col 3: k=3, i=2 -> 3*3+2 = 11.
    # C2, col 4: k=4, i=2 -> 4*3+2 = 14.
    
    # Now we construct the blocks.
    
    # Block 1: A1_top * B_top_left (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    u_idx = [0, 1, 4, 5]
    v_idx = [0, 1, 5, 6]
    w_idx = [0, 3, 6, 9, 12] # Wait, W is 2x2. Indices are C00, C01, C10, C11.
    # C1_top (row 0) and C1_bottom (row 1).
    # C1_top, cols 0,1 -> W indices 0, 3.
    # C1_bottom, cols 0,1 -> W indices 1, 4.
    # So W indices for this block are 0, 3, 1, 4.
    w_idx = [0, 3, 1, 4]
    
    for r in range(7):
        U_parts.append(np.zeros(n * m))
        for i, idx in enumerate(u_idx):
            U_parts[-1][idx] = U_blk[i, r]
        
        V_parts.append(np.zeros(m * p))
        for i, idx in enumerate(v_idx):
            V_parts[-1][idx] = V_blk[i, r]
        
        W_parts.append(np.zeros(p * n))
        for i, idx in enumerate(w_idx):
            W_parts[-1][idx] = W_blk[i, r]
    
    # Block 2: A1_top * B_top_mid (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    u_idx = [0, 1, 4, 5]
    v_idx = [2, 3, 7, 8]
    w_idx = [6, 9, 7, 10] # C1_top cols 2,3 -> 6,9. C1_bottom cols 2,3 -> 7,10.
    
    for r in range(7):
        U_parts.append(np.zeros(n * m))
        for i, idx in enumerate(u_idx):
            U_parts[-1][idx] = U_blk[i, r]
        
        V_parts.append(np.zeros(m * p))
        for i, idx in enumerate(v_idx):
            V_parts[-1][idx] = V_blk[i, r]
        
        W_parts.append(np.zeros(p * n))
        for i, idx in enumerate(w_idx):
            W_parts[-1][idx] = W_blk[i, r]
    
    # Block 3: A1_top * B_top_right (2x2x1)
    U_blk, V_blk, W_blk = schoolbook_2x2x1()
    u_idx = [0, 1, 4, 5]
    v_idx = [4, 9]
    w_idx = [12, 13] # C1_top col 4 -> 12. C1_bottom col 4 -> 13.
    
    for r in range(4):
        U_parts.append(np.zeros(n * m))
        for i, idx in enumerate(u_idx):
            U_parts[-1][idx] = U_blk[i, r]
        
        V_parts.append(np.zeros(m * p))
        for i, idx in enumerate(v_idx):
            V_parts[-1][idx] = V_blk[i, r]
        
        W_parts.append(np.zeros(p * n))
        for i, idx in enumerate(w_idx):
            W_parts[-1][idx] = W_blk[i, r]
    
    # Block 4: A1_bottom * B_bottom_left (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    u_idx = [2, 3, 6, 7]
    v_idx = [10, 11, 15, 16]
    w_idx = [0, 3, 1, 4]
    
    for r in range(7):
        U_parts.append(np.zeros(n * m))
        for i, idx in enumerate(u_idx):
            U_parts[-1][idx] = U_blk[i, r]
        
        V_parts.append(np.zeros(m * p))
        for i, idx in enumerate(v_idx):
            V_parts[-1][idx] = V_blk[i, r]
        
        W_parts.append(np.zeros(p * n))
        for i, idx in enumerate(w_idx):
            W_parts[-1][idx] = W_blk[i, r]
    
    # Block 5: A1_bottom * B_bottom_mid (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    u_idx = [2, 3, 6, 7]
    v_idx = [12, 13, 17, 18]
    w_idx = [6, 9, 7, 10]
    
    for r in range(7):
        U_parts.append(np.zeros(n * m))
        for i, idx in enumerate(u_idx):
            U_parts[-1][idx] = U_blk[i, r]
        
        V_parts.append(np.zeros(m * p))
        for i, idx in enumerate(v_idx):
            V_parts[-1][idx] = V_blk[i, r]
        
        W_parts.append(np.zeros(p * n))
        for i, idx in enumerate(w_idx):
            W_parts[-1][idx] = W_blk[i, r]
    
    # Block 6: A1_bottom * B_bottom_right (2x2x1)
    U_blk, V_blk, W_blk = schoolbook_2x2x1()
    u_idx = [2, 3, 6, 7]
    v_idx = [14, 19]
    w_idx = [12, 13]
    
    for r in range(4):
        U_parts.append(np.zeros(n * m))
        for i, idx in enumerate(u_idx):
            U_parts[-1][idx] = U_blk[i, r]
        
        V_parts.append(np.zeros(m * p))
        for i, idx in enumerate(v_idx):
            V_parts[-1][idx] = V_blk[i, r]
        
        W_parts.append(np.zeros(p * n))
        for i, idx in enumerate(w_idx):
            W_parts[-1][idx] = W_blk[i, r]
    
    # Part 2: A2 (row 2) * B
    # A2_top: row 2; cols 0,1. Indices: 8,9.
    # A2_bottom: row 2; cols 2,3. Indices: 10,11.
    
    # Block 7: A2_top * B_top_left (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    u_idx = [8, 9]
    v_idx = [0, 1, 5, 6]
    w_idx = [2, 5] # C2 cols 0,1 -> 2,5.
    
    for r in range(4):
        U_parts.append(np.zeros(n * m))
        for i, idx in enumerate(u_idx):
            U_parts[-1][idx] = U_blk[i, r]
        
        V_parts.append(np.zeros(m * p))
        for i, idx in enumerate(v_idx):
            V_parts[-1][idx] = V_blk[i, r]
        
        W_parts.append(np.zeros(p * n))
        for i, idx in enumerate(w_idx):
            W_parts[-1][idx] = W_blk[i, r]
    
    # Block 8: A2_top * B_top_mid (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    u_idx = [8, 9]
    v_idx = [2, 3, 7, 8]
    w_idx = [8, 11] # C2 cols 2,3 -> 8,11.
    
    for r in range(4):
        U_parts.append(np.zeros(n * m))
        for i, idx in enumerate(u_idx):
            U_parts[-1][idx] = U_blk[i, r]
        
        V_parts.append(np.zeros(m * p))
        for i, idx in enumerate(v_idx):
            V_parts[-1][idx] = V_blk[i, r]
        
        W_parts.append(np.zeros(p * n))
        for i, idx in enumerate(w_idx):
            W_parts[-1][idx] = W_blk[i, r]
    
    # Block 9: A2_top * B_top_right (1x2x1)
    U_blk, V_blk, W_blk = schoolbook_1x2x1()
    u_idx = [8, 9]
    v_idx = [4, 9]
    w_idx = [14] # C2 col 4 -> 14.
    
    for r in range(2):
        U_parts.append(np.zeros(n * m))
        for i, idx in enumerate(u_idx):
            U_parts[-1][idx] = U_blk[i, r]
        
        V_parts.append(np.zeros(m * p))
        for i, idx in enumerate(v_idx):
            V_parts[-1][idx] = V_blk[i, r]
        
        W_parts.append(np.zeros(p * n))
        for i, idx in enumerate(w_idx):
            W_parts[-1][idx] = W_blk[i, r]
    
    # Block 10: A2_bottom * B_bottom_left (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    u_idx = [10, 11]
    v_idx = [10, 11, 15, 16]
    w_idx = [2, 5]
    
    for r in range(4):
        U_parts.append(np.zeros(n * m))
        for i, idx in enumerate(u_idx):
            U_parts[-1][idx] = U_blk[i, r]
        
        V_parts.append(np.zeros(m * p))
        for i, idx in enumerate(v_idx):
            V_parts[-1][idx] = V_blk[i, r]
        
        W_parts.append(np.zeros(p * n))
        for i, idx in enumerate(w_idx):
            W_parts[-1][idx] = W_blk[i, r]
    
    # Block 11: A2_bottom * B_bottom_mid (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    u_idx = [10, 11]
    v_idx = [12, 13, 17, 18]
    w_idx = [8, 11]
    
    for r in range(4):
        U_parts.append(np.zeros(n * m))
        for i, idx in enumerate(u_idx):
            U_parts[-1][idx] = U_blk[i, r]
        
        V_parts.append(np.zeros(m * p))
        for i, idx in enumerate(v_idx):
            V_parts[-1][idx] = V_blk[i, r]
        
        W_parts.append(np.zeros(p * n))
        for i, idx in enumerate(w_idx):
            W_parts[-1][idx] = W_blk[i, r]
    
    # Block 12: A2_bottom * B_bottom_right (1x2x1)
    U_blk, V_blk, W_blk = schoolbook_1x2x1()
    u_idx = [10, 11]
    v_idx = [14, 19]
    w_idx = [14]
    
    for r in range(2):
        U_parts.append(np.zeros(n * m))
        for i, idx in enumerate(u_idx):
            U_parts[-1][idx] = U_blk[i, r]
        
        V_parts.append(np.zeros(m * p))
        for i, idx in enumerate(v_idx):
            V_parts[-1][idx] = V_blk[i, r]
        
        W_parts.append(np.zeros(p * n))
        for i, idx in enumerate(w_idx):
            W_parts[-1][idx] = W_blk[i, r]
    
    # Total rank: 7+7+4+7+7+4+4+4+2+4+4+2 = 56.
    # Wait, 1x2x2 is rank 4?
    # A (1x2) * B (2x2) = C (1x2).
    # C0 = A0*B00 + A1*B10
    # C1 = A0*B01 + A1*B11
    # This is 4 multiplications. Rank 4.
    # Can we do better?
    # Strassen for 1x2x2? No, Strassen is for 2x2x2.
    # Is there a rank 3 decomposition for 1x2x2?
    # It's a bilinear map from R^2 x R^4 to R^2.
    # Rank is at least 2.
    # Let's check if rank 3 is possible.
    # If rank 3, then 3 multiplications.
    # M1 = (A0+A1)B00
    # M2 = A0(B01-B11)
    # M3 = A1(B10+B11)
    # C0 = M1 + M3 - M2? No.
    # Let's just use rank 4 for now. It's safe.
    # Total rank = 56.
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    # Verify the decomposition is correct using vectorized computation
    if time.time() < _DEADLINE - 5:  # Leave 5s margin
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_test = A_test @ B_test
        
        # Vectorized verification:
        # C[i,k] = sum_r W[k*n+i, r] * sum_j U[i*m+j, r] * V[j*p+k, r]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_reshaped = U.reshape(n, m, -1)  # (n, m, rank)
        V_reshaped = V.reshape(m, p, -1)  # (m, p, rank)
        W_reshaped = W.reshape(p, n, -1)  # (p, n, rank)
        
        # For each r: sum_j U[i,j,r] * V[j,k,r] gives a (n, p) matrix
        # Then multiply by W[k,i,r] element-wise and sum over r
        # Using einsum: C[i,k] = sum_r sum_j U[i,j,r] * V[j,k,r] * W[k,i,r]
        C_decomp = np.einsum('ijr,jkr,kir->ik', U_reshaped, V_reshaped, W_reshaped)
        
        if not np.allclose(C_decomp, C_test, atol=1e-10):
            return _schoolbook_fallback(n, m, p)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _schoolbook_fallback(n, m, p):
    """Schoolbook decomposition with rank n*m*p."""
    U_parts = []
    V_parts = []
    W_parts = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n * m)
                u[i * m + j] = 1
                v = np.zeros(m * p)
                v[j * p + k] = 1
                w = np.zeros(p * n)
                w[k * n + i] = 1
                U_parts.append(u)
                V_parts.append(v)
                W_parts.append(w)
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
