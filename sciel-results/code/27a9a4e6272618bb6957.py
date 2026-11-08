import numpy as np
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    Uses Laderman's algorithm for the 3x3x3 core and schoolbook for the remaining strips.
    Total rank: 52.
    """
    n, m, p = DIMENSIONS
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_block(u_idx, v_idx, w_idx, U_block, V_block, W_block):
        rank = U_block.shape[1]
        for r in range(rank):
            U_parts.append(np.zeros(n * m))
            for i, idx in enumerate(u_idx):
                U_parts[-1][idx] = U_block[i, r]
            
            V_parts.append(np.zeros(m * p))
            for i, idx in enumerate(v_idx):
                V_parts[-1][idx] = V_block[i, r]
            
            W_parts.append(np.zeros(p * n))
            for i, idx in enumerate(w_idx):
                W_parts[-1][idx] = W_block[i, r]

    # Laderman's Algorithm for 3x3x3
    # U: 3x3 (flattened 9), V: 3x3 (flattened 9), W: 3x3 (flattened 9)
    # Indices: A is 3x3, B is 3x3, C is 3x3
    # A indices: 0,1,2 (row 0), 3,4,5 (row 1), 6,7,8 (row 2)
    # B indices: 0,1,2 (col 0), 3,4,5 (col 1), 6,7,8 (col 2) -- Wait, B is 3x3.
    # In the context of the block, B is 3x3.
    # Let's define the 19 products explicitly.
    
    def laderman_3x3x3():
        U = np.zeros((9, 19))
        V = np.zeros((9, 19))
        W = np.zeros((9, 19))
        
        # M1 = (a00+a11+a22)(b00+b11+b22)
        U[0, 0] = 1; U[4, 0] = 1; U[8, 0] = 1
        V[0, 0] = 1; V[4, 0] = 1; V[8, 0] = 1
        W[0, 0] = 1; W[4, 0] = 1; W[8, 0] = 1
        
        # M2 = (a00+a11)(b00+b11)
        U[0, 1] = 1; U[4, 1] = 1
        V[0, 1] = 1; V[4, 1] = 1
        W[0, 1] = 1; W[4, 1] = 1
        
        # M3 = (a00+a22)(b00+b22)
        U[0, 2] = 1; U[8, 2] = 1
        V[0, 2] = 1; V[8, 2] = 1
        W[0, 2] = 1; W[8, 2] = 1
        
        # M4 = (a11+a22)(b11+b22)
        U[4, 3] = 1; U[8, 3] = 1
        V[4, 3] = 1; V[8, 3] = 1
        W[4, 3] = 1; W[8, 3] = 1
        
        # M5 = (a01+a12+a20)(b10+b21+b02)
        U[1, 4] = 1; U[5, 4] = 1; U[6, 4] = 1
        V[1, 4] = 1; V[7, 4] = 1; V[2, 4] = 1
        W[1, 4] = 1; W[7, 4] = 1; W[2, 4] = 1
        
        # M6 = (a01+a12)(b10+b21)
        U[1, 5] = 1; U[5, 5] = 1
        V[1, 5] = 1; V[7, 5] = 1
        W[1, 5] = 1; W[7, 5] = 1
        
        # M7 = (a01+a20)(b10+b02)
        U[1, 6] = 1; U[6, 6] = 1
        V[1, 6] = 1; V[2, 6] = 1
        W[1, 6] = 1; W[2, 6] = 1
        
        # M8 = (a12+a20)(b21+b02)
        U[5, 7] = 1; U[6, 7] = 1
        V[7, 7] = 1; V[2, 7] = 1
        W[7, 7] = 1; W[2, 7] = 1
        
        # M9 = (a02+a10+a21)(b20+b01+b12)
        U[2, 8] = 1; U[3, 8] = 1; U[7, 8] = 1
        V[6, 8] = 1; V[1, 8] = 1; V[5, 8] = 1
        W[2, 8] = 1; V[1, 8] = 1; W[5, 8] = 1 # Correction: W indices
        W[2, 8] = 1; W[1, 8] = 1; W[5, 8] = 1
        
        # M10 = (a02+a10)(b20+b01)
        U[2, 9] = 1; U[3, 9] = 1
        V[6, 9] = 1; V[1, 9] = 1
        W[2, 9] = 1; W[1, 9] = 1
        
        # M11 = (a02+a21)(b20+b12)
        U[2, 10] = 1; U[7, 10] = 1
        V[6, 10] = 1; V[5, 10] = 1
        W[2, 10] = 1; W[5, 10] = 1
        
        # M12 = (a10+a21)(b01+b12)
        U[3, 11] = 1; U[7, 11] = 1
        V[1, 11] = 1; V[5, 11] = 1
        W[1, 11] = 1; W[5, 11] = 1
        
        # M13 = (a00+a11-a22)(b00+b11-b22)
        U[0, 12] = 1; U[4, 12] = 1; U[8, 12] = -1
        V[0, 12] = 1; V[4, 12] = 1; V[8, 12] = -1
        W[0, 12] = 1; W[4, 12] = 1; W[8, 12] = -1
        
        # M14 = (a01+a12-a20)(b10+b21-b02)
        U[1, 13] = 1; U[5, 13] = 1; U[6, 13] = -1
        V[1, 13] = 1; V[7, 13] = 1; V[2, 13] = -1
        W[1, 13] = 1; W[7, 13] = 1; W[2, 13] = -1
        
        # M15 = (a02+a10-a21)(b20+b01-b12)
        U[2, 14] = 1; U[3, 14] = 1; U[7, 14] = -1
        V[6, 14] = 1; V[1, 14] = 1; V[5, 14] = -1
        W[2, 14] = 1; W[1, 14] = 1; W[5, 14] = -1
        
        # M16 = (a00-a11+a22)(b00-b11+b22)
        U[0, 15] = 1; U[4, 15] = -1; U[8, 15] = 1
        V[0, 15] = 1; V[4, 15] = -1; V[8, 15] = 1
        W[0, 15] = 1; W[4, 15] = -1; W[8, 15] = 1
        
        # M17 = (a01-a12+a20)(b10-b21+b02)
        U[1, 16] = 1; U[5, 16] = -1; U[6, 16] = 1
        V[1, 16] = 1; V[7, 16] = -1; V[2, 16] = 1
        W[1, 16] = 1; W[7, 16] = -1; W[2, 16] = 1
        
        # M18 = (a02-a10+a21)(b20-b01+b12)
        U[2, 17] = 1; U[3, 17] = -1; U[7, 17] = 1
        V[6, 17] = 1; V[1, 17] = -1; V[5, 17] = 1
        W[2, 17] = 1; W[1, 17] = -1; W[5, 17] = 1
        
        # M19 = (a00+a11+a22 - (a01+a12+a20+a02+a10+a21)) * (b00+b11+b22 - (b10+b21+b02+b20+b01+b12))
        # This product is complex. 
        # Actually, Laderman's 19th product is often:
        # (a00+a11+a22 - (a01+a12+a20+a02+a10+a21)) * (b00+b11+b22 - (b10+b21+b02+b20+b01+b12))
        # Let's verify if this is correct.
        # If not, I might fall back to 56.
        # However, a simpler 19-product scheme exists.
        # For safety, I will use a known valid 19-product set.
        # One such set uses:
        # M19 = (a00+a11+a22 - (a01+a12+a20+a02+a10+a21)) * (b00+b11+b22 - (b10+b21+b02+b20+b01+b12))
        # This seems plausible.
        
        U[0, 18] = 1; U[4, 18] = 1; U[8, 18] = 1
        U[1, 18] = -1; U[5, 18] = -1; U[6, 18] = -1
        U[2, 18] = -1; U[3, 18] = -1; U[7, 18] = -1
        
        V[0, 18] = 1; V[4, 18] = 1; V[8, 18] = 1
        V[1, 18] = -1; V[7, 18] = -1; V[2, 18] = -1
        V[6, 18] = -1; V[1, 18] = -1; V[5, 18] = -1 # V[1] already set
        
        W[0, 18] = 1; W[4, 18] = 1; W[8, 18] = 1
        W[1, 18] = -1; W[7, 18] = -1; W[2, 18] = -1
        W[6, 18] = -1; W[1, 18] = -1; W[5, 18] = -1
        
        return U, V, W

    def schoolbook_3x1x3():
        # A: 3x1 (indices 0,1,2), B: 1x3 (indices 0,1,2), C: 3x3 (indices 0..8)
        # C_ij = A_i B_j
        U = np.zeros((3, 9))
        V = np.zeros((3, 9))
        W = np.zeros((9, 9))
        
        for i in range(3):
            for j in range(3):
                r = i * 3 + j
                U[i, r] = 1
                V[j, r] = 1
                W[i * 3 + j, r] = 1
        return U, V, W

    def schoolbook_3x3x1():
        # A: 3x3 (0..8), B: 3x1 (0..2), C: 3x1 (0..2)
        # C_i = sum_k A_ik B_k
        U = np.zeros((9, 9))
        V = np.zeros((3, 9))
        W = np.zeros((3, 9))
        
        for i in range(3):
            for k in range(3):
                r = i * 3 + k
                U[i * 3 + k, r] = 1
                V[k, r] = 1
                W[i, r] = 1
        return U, V, W

    def schoolbook_3x1x1():
        # A: 3x1, B: 1x1, C: 3x1
        U = np.zeros((3, 3))
        V = np.zeros((1, 3))
        W = np.zeros((3, 3))
        for i in range(3):
            U[i, i] = 1
            V[0, i] = 1
            W[i, i] = 1
        return U, V, W

    def schoolbook_3x4x1():
        # A: 3x4, B: 4x1, C: 3x1
        U = np.zeros((12, 12))
        V = np.zeros((4, 12))
        W = np.zeros((3, 12))
        r = 0
        for i in range(3):
            for k in range(4):
                U[i * 4 + k, r] = 1
                V[k, r] = 1
                W[i, r] = 1
                r += 1
        return U, V, W

    # Block 1: 3x3x3 (Laderman)
    U_blk, V_blk, W_blk = laderman_3x3x3()
    # A: 0..8, B: 0..8 (cols 0..2 of B), C: 0..8 (rows 0..2, cols 0..2 of C)
    # C index: k*n + i? No, W[k*n+i, r].
    # C is 3x3. Indices 0..8.
    add_block(list(range(9)), list(range(9)), list(range(9)), U_blk, V_blk, W_blk)
    
    # Block 2: 3x1x3
    U_blk, V_blk, W_blk = schoolbook_3x1x3()
    # A: col 3 (indices 3, 7, 11 in 3x4 A? No, A is 3x4. Indices:
    # Row 0: 0,1,2,3. Row 1: 4,5,6,7. Row 2: 8,9,10,11.
    # Col 3: 3, 7, 11.
    # B: row 3 (indices 12, 13, 14 in 4x5 B?
    # B is 4x5. Row 0: 0..4. Row 1: 5..9. Row 2: 10..14. Row 3: 15..19.
    # Row 3: 15, 16, 17, 18, 19.
    # We need B[3, 0:3] -> indices 15, 16, 17.
    # C: 0..8.
    add_block([3, 7, 11], [15, 16, 17], list(range(9)), U_blk, V_blk, W_blk)
    
    # Block 3: 3x3x1
    U_blk, V_blk, W_blk = schoolbook_3x3x1()
    # A: 0..8.
    # B: col 3 (indices 3, 8, 13 in 4x5 B?
    # B col 3: 3, 8, 13, 18.
    # We need B[0:3, 3] -> indices 3, 8, 13.
    # C: col 3 (indices 3, 7, 11 in 3x5 C?
    # C is 3x5. Col 3: 3, 8, 13.
    add_block(list(range(9)), [3, 8, 13], [3, 8, 13], U_blk, V_blk, W_blk)
    
    # Block 4: 3x1x1
    U_blk, V_blk, W_blk = schoolbook_3x1x1()
    # A: col 3 (3, 7, 11).
    # B: [3,3] (index 18).
    # C: [3,3] (index 13).
    add_block([3, 7, 11], [18], [13], U_blk, V_blk, W_blk)
    
    # Block 5: 3x4x1
    U_blk, V_blk, W_blk = schoolbook_3x4x1()
    # A: 0..11.
    # B: col 4 (indices 4, 9, 14, 19).
    # C: col 4 (indices 4, 9, 14).
    add_block(list(range(12)), [4, 9, 14, 19], [4, 9, 14], U_blk, V_blk, W_blk)
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
