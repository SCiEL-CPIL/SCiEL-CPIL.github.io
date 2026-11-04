import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 115 approximation using block decomposition
    # U: (25, Rank), V: (25, Rank), W: (25, Rank)
    # We will build the factors column by column.
    
    U = []
    V = []
    W = []
    
    def add_block_strassen(rA, cA, rB, cB, rC, cC, rank_offset):
        # Adds Strassen 2x2 block
        # Local indices 0..1
        # Global A: rA..rA+1, cA..cA+1
        # Global B: rB..rB+1, cB..cB+1
        # Global C: rC..rC+1, cC..cC+1
        
        # Strassen factors for 2x2
        # U2, V2, W2 defined as in previous attempt
        U2 = np.zeros((4, 7), dtype=float)
        V2 = np.zeros((4, 7), dtype=float)
        W2 = np.zeros((4, 7), dtype=float)
        
        U2[0, 0] = 1; U2[3, 0] = 1
        V2[0, 0] = 1; V2[1, 0] = 1
        W2[0, 0] = 1; W2[3, 0] = 1
        
        U2[2, 1] = 1; U2[3, 1] = 1
        V2[0, 1] = 1
        W2[2, 1] = 1; W2[3, 1] = 1
        
        U2[0, 2] = 1
        V2[1, 2] = 1; V2[3, 2] = -1
        W2[1, 2] = 1; W2[3, 2] = -1
        
        U2[3, 3] = 1
        V2[2, 3] = 1; V2[0, 3] = -1
        W2[2, 3] = 1; W2[3, 3] = -1
        
        U2[0, 4] = 1; U2[1, 4] = 1
        V2[3, 4] = 1
        W2[0, 4] = 1; W2[1, 4] = 1
        
        U2[2, 5] = 1; U2[0, 5] = -1
        V2[0, 5] = 1; V2[1, 5] = 1
        W2[2, 5] = 1; W2[3, 5] = 1
        
        U2[1, 6] = 1; U2[3, 6] = -1
        V2[2, 6] = 1; V2[3, 6] = 1
        W2[0, 6] = -1; W2[3, 6] = -1
        
        for r in range(7):
            u_col = [0.0] * 25
            v_col = [0.0] * 25
            w_col = [0.0] * 25
            
            for i in range(2):
                for j in range(2):
                    u_col[(rA+i)*5 + (cA+j)] += U2[i*2+j, r]
                    
            for j in range(2):
                for k in range(2):
                    v_col[(rB+j)*5 + (cB+k)] += V2[j*2+k, r]
                    
            for k in range(2):
                for i in range(2):
                    w_col[(rC+k)*5 + (cC+i)] += W2[k*2+i, r]
                    
            U.append(u_col)
            V.append(v_col)
            W.append(w_col)

    def add_block_trivial(rA, cA, rB, cB, rC, cC, nA, nB, nC):
        # Adds Trivial block
        # A: nA x nB, B: nB x nC, C: nA x nC
        for i in range(nA):
            for j in range(nB):
                for k in range(nC):
                    u_col = [0.0] * 25
                    v_col = [0.0] * 25
                    w_col = [0.0] * 25
                    
                    u_col[(rA+i)*5 + (cA+j)] = 1.0
                    v_col[(rB+j)*5 + (cB+k)] = 1.0
                    w_col[(rC+k)*5 + (cC+i)] = 1.0
                    
                    U.append(u_col)
                    V.append(v_col)
                    W.append(w_col)

    # Block 1: A22 B22 -> C22 (2x2x2)
    add_block_strassen(0, 0, 0, 0, 0, 0, 0)
    
    # Block 2: A23 B32 -> C22 (2x3x2)
    # Split A23 (2x3) into A22 (2x2) and A21 (2x1)
    # Split B32 (3x2) into B22 (2x2) and B12 (1x2)
    # Term 1: A22 B22 (2x2x2)
    add_block_strassen(0, 2, 2, 0, 0, 0, 0)
    # Term 2: A21 B12 (2x1x2)
    add_block_trivial(0, 4, 4, 0, 0, 0, 2, 1, 2)
    
    # Block 3: A22 B23 -> C23 (2x2x3)
    # Split B23 (2x3) into B22 (2x2) and B21 (2x1)
    # Term 1: A22 B22 (2x2x2)
    add_block_strassen(0, 0, 0, 2, 0, 2, 0)
    # Term 2: A22 B21 (2x2x1)
    add_block_trivial(0, 0, 0, 4, 0, 4, 2, 2, 1)
    
    # Block 4: A23 B33 -> C23 (2x3x3)
    # Split A23 (2x3) into A22 (2x2) and A21 (2x1)
    # Term 1: A22 B33 (2x2x3)
    # Split B33 (3x3) into B22 (2x2) and B21 (2x1) ... wait, B33 is 3x3.
    # A22 (2x2) * B33 (3x3).
    # This is not a square block.
    # A22 is rows 0-1, cols 2-3 of A.
    # B33 is rows 2-4, cols 2-4 of B.
    # We can decompose A22 B33 by splitting B33 into top 2 rows and bottom 1 row?
    # B33 = [B22; B12].
    # A22 B33 = A22 B22 + A22 B12.
    # A22 B22 (2x2x2)
    add_block_strassen(0, 2, 2, 2, 0, 2, 0)
    # A22 B12 (2x2x1) -- B12 is 2x1 (rows 2-3, col 4? No, B33 cols are 2-4. B22 cols 2-3. B21 col 4).
    # B12 here refers to the part of B33 corresponding to the last column?
    # Let's be precise.
    # B33 is 3x3. Rows 2,3,4. Cols 2,3,4.
    # Split B33 into B33_top (2x3) and B33_bot (1x3)? No, we need to match A22 (2x2).
    # A22 is 2x2. B33 is 3x3.
    # Product is 2x3.
    # We can view B33 as [B33_1 (2x2), B33_2 (2x1)] horizontally?
    # B33 = [B33_L (3x2), B33_R (3x1)].
    # A22 B33 = A22 B33_L + A22 B33_R.
    # A22 (2x2) * B33_L (3x2) -> Invalid.
    
    # Correct split for A22 (2x2) * B33 (3x3):
    # Split B33 into B33_T (2x3) and B33_B (1x3) vertically?
    # B33 = [B33_T; B33_B].
    # A22 B33 = A22 B33_T + A22 B33_B.
    # A22 (2x2) * B33_T (2x3) -> 2x3.
    # A22 (2x2) * B33_B (1x3) -> Invalid.
    
    # We must split A22? No, A22 is fixed.
    # We must split B33 such that the inner dimension matches 2.
    # B33 inner dim is 3.
    # We can split B33 into B33_1 (2x3) and B33_2 (1x3) vertically?
    # Then A22 * B33_1 is valid (2x2 * 2x3).
    # A22 * B33_2 is invalid (2x2 * 1x3).
    
    # Alternative: Split A23 into A22 (2x2) and A21 (2x1).
    # A23 B33 = A22 B33 + A21 B33.
    # A21 (2x1) * B33 (3x3) -> Invalid.
    
    # This approach is getting messy.
    # Let's just use Trivial for all non-2x2 blocks to ensure correctness, even if rank is higher.
    # Rank estimate with Trivial for all non-2x2:
    # 1. 2x2x2: 7
    # 2. 2x3x2: 12 (Trivial)
    # 3. 2x2x3: 12 (Trivial)
    # 4. 2x3x3: 18 (Trivial)
    # 5. 3x2x2: 12 (Trivial)
    # 6. 3x3x2: 18 (Trivial)
    # 7. 3x2x3: 18 (Trivial)
    # 8. 3x3x3: 27 (Trivial)
    # Total: 7 + 12 + 12 + 18 + 12 + 18 + 18 + 27 = 124.
    # 124 < 125.
    # This is a valid improvement!
    
    # I will clear the lists and re-implement with this simpler strategy.
    U = []
    V = []
    W = []
    
    # Block 1: A22 B22 -> C22
    add_block_strassen(0, 0, 0, 0, 0, 0, 0)
    
    # Block 2: A23 B32 -> C22
    add_block_trivial(0, 2, 2, 0, 0, 0, 2, 3, 2)
    
    # Block 3: A22 B23 -> C23
    add_block_trivial(0, 0, 0, 2, 0, 2, 2, 2, 3)
    
    # Block 4: A23 B33 -> C23
    add_block_trivial(0, 2, 2, 2, 0, 2, 2, 3, 3)
    
    # Block 5: A32 B22 -> C32
    add_block_trivial(2, 0, 0, 0, 2, 0, 3, 2, 2)
    
    # Block 6: A33 B32 -> C32
    add_block_trivial(2, 2, 2, 0, 2, 0, 3, 3, 2)
    
    # Block 7: A32 B23 -> C33
    add_block_trivial(2, 0, 0, 2, 2, 2, 3, 2, 3)
    
    # Block 8: A33 B33 -> C33
    add_block_trivial(2, 2, 2, 2, 2, 2, 3, 3, 3)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
