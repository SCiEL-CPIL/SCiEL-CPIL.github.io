import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    Uses a block-diagonal approach:
    - Split 3x4x5 into (2+1)x4x(2+2+1) blocks
    - Use Strassen for 2x2x2 blocks (rank 7)
    - Use schoolbook for 2x2x1 (rank 4) and 1x2x2 (rank 4) and 1x2x1 (rank 2)
    Total rank: 56
    """
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=3, m=4, p=5
    
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

    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        
        return U, V, W

    def schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        
        return U, V, W

    def schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        
        return U, V, W

    # Check if we're still within time budget
    if time.time() > deadline:
        return {
            "U": np.zeros((n*m, 1)).tolist(),
            "V": np.zeros((m*p, 1)).tolist(),
            "W": np.zeros((p*n, 1)).tolist()
        }

    # Block 1: A rows 0-1, B cols 0-1 (2x2x2) -> W cols 0, 3; W rows 0, 1
    U_blk, V_blk, W_blk = strassen_2x2x2()
    # u_idx: rows of A: 0,1,4,5 (row-major for 3x4 matrix: [0,1,2,3, 4,5,6,7, 8,9,10,11])
    # Actually n=3, m=4, so A is 3x4, index = i*m + j
    # Row 0: 0,1,2,3
    # Row 1: 4,5,6,7
    # Row 2: 8,9,10,11
    # B is 4x5, index = i*p + j
    # Row 0: 0,1,2,3,4
    # Row 1: 5,6,7,8,9
    # Row 2: 10,11,12,13,14
    # Row 3: 15,16,17,18,19
    # W is 5x3, index = k*n + i (trace convention W[k*n+i, r])
    # Col 0: 0,1,2,3,4 (k=0: 0,1,2; k=1: 3,4,5; etc.)
    # Actually W[k*n+i] where k is column of B (0..4), i is row of A (0..2)
    # k=0: 0,1,2
    # k=1: 3,4,5
    # k=2: 6,7,8
    # k=3: 9,10,11
    # k=4: 12,13,14
    
    # For 2x2x2 block: A rows [r1,r2], B cols [c1,c2], W: k in [c1,c2], i in [r1,r2]
    # u_idx: A indices for rows r1,r2 and cols 0,1 (first 2 cols of B)
    # For A row r1: indices r1*4+0, r1*4+1
    # For A row r2: indices r2*4+0, r2*4+1
    
    # Block 1: A rows 0,1; B cols 0,1; W k=0,1; i=0,1
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block([0, 1, 4, 5], [0, 1, 5, 6], [0, 1, 3, 4], U_blk, V_blk, W_blk)
    
    if time.time() > deadline:
        return {
            "U": np.zeros((n*m, 1)).tolist(),
            "V": np.zeros((m*p, 1)).tolist(),
            "W": np.zeros((p*n, 1)).tolist()
        }
    
    # Block 2: A rows 0,1; B cols 2,3; W k=2,3; i=0,1
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block([0, 1, 4, 5], [2, 3, 7, 8], [6, 7, 9, 10], U_blk, V_blk, W_blk)
    
    if time.time() > deadline:
        return {
            "U": np.zeros((n*m, 1)).tolist(),
            "V": np.zeros((m*p, 1)).tolist(),
            "W": np.zeros((p*n, 1)).tolist()
        }
    
    # Block 3: A rows 0,1; B col 4 (1 col); W k=4; i=0,1 (2x2x1)
    U_blk, V_blk, W_blk = schoolbook_2x2x1()
    add_block([0, 1, 4, 5], [4, 9], [12, 13], U_blk, V_blk, W_blk)
    
    if time.time() > deadline:
        return {
            "U": np.zeros((n*m, 1)).tolist(),
            "V": np.zeros((m*p, 1)).tolist(),
            "W": np.zeros((p*n, 1)).tolist()
        }
    
    # Block 4: A rows 2,3; B cols 0,1; W k=0,1; i=2,3
    # Wait, A only has 3 rows (0,1,2). Let me reconsider the split.
    # n=3, so rows are 0,1,2.
    # Let me redo: split A into rows [0,1] and row [2]
    # Split B cols into [0,1], [2,3], [4]
    # Split W k into [0,1], [2,3], [4]
    
    # Actually let me restart with clearer indexing.
    # n=3 (rows of A), m=4 (cols of A = rows of B), p=5 (cols of B)
    # A: 3x4, B: 4x5, C = A*B: 3x5
    # W[k*n + i] where k = col of C (= col of B), i = row of C (= row of A)
    
    # U has n*m = 12 entries, indexed by (i,j) -> i*4+j
    # V has m*p = 20 entries, indexed by (j,k) -> j*5+k
    # W has p*n = 15 entries, indexed by (k,i) -> k*3+i
    
    # Split: A rows {0,1} and {2}, B cols {0,1}, {2,3}, {4}
    # This gives blocks:
    # (rows {0,1}, cols {0,1}) -> 2x2x2, rank 7
    # (rows {0,1}, cols {2,3}) -> 2x2x2, rank 7
    # (rows {0,1}, col {4})    -> 2x2x1, rank 4
    # (row {2}, cols {0,1})    -> 1x2x2, rank 4
    # (row {2}, cols {2,3})    -> 1x2x2, rank 4
    # (row {2}, col {4})       -> 1x2x1, rank 2
    # Total: 7+7+4+4+4+2 = 28? No wait.
    
    # For 2x2x2: U has 4 entries (2 rows x 2 cols of A), V has 4 entries (2 rows x 2 cols of B), W has 4 entries
    # For 2x2x1: U has 4 entries, V has 2 entries (2 rows x 1 col), W has 2 entries
    # For 1x2x2: U has 2 entries (1 row x 2 cols), V has 4 entries, W has 2 entries
    # For 1x2x1: U has 2 entries, V has 2 entries, W has 1 entry
    
    # But the ranks are: 2x2x2 -> 7, 2x2x1 -> 4, 1x2x2 -> 4, 1x2x1 -> 2
    # Total: 7+7+4+4+4+2 = 28? That seems too low. Let me re-examine.
    
    # Actually for matrix multiplication tensor M_{n,m,p}, the rank for 2x2x2 is 7 (Strassen)
    # For 2x2x1: this is multiplying 2x2 by 2x1, which is 4 products in schoolbook, rank 4
    # For 1x2x2: 1x2 by 2x2, rank 4 in schoolbook
    # For 1x2x1: 1x2 by 2x1, rank 2 in schoolbook
    # Total: 7+7+4+4+4+2 = 28
    
    # But the original code had 56. Let me re-read the original.
    # The original splits differently. Let me just keep the original block structure.
    
    # Let me restart and just use the original structure which gave rank 56.
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # The original code structure - let me replicate it exactly
    # Block 1: A1_top * B_top_left (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block([0, 1, 4, 5], [0, 1, 5, 6], [0, 3, 1, 4], U_blk, V_blk, W_blk)
    
    # Block 2: A1_top * B_top_mid (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block([0, 1, 4, 5], [2, 3, 7, 8], [6, 9, 7, 10], U_blk, V_blk, W_blk)
    
    # Block 3: A1_top * B_top_right (2x2x1)
    U_blk, V_blk, W_blk = schoolbook_2x2x1()
    add_block([0, 1, 4, 5], [4, 9], [12, 13], U_blk, V_blk, W_blk)
    
    # Block 4: A1_bottom * B_bottom_left (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block([2, 3, 6, 7], [10, 11, 15, 16], [0, 3, 1, 4], U_blk, V_blk, W_blk)
    
    # Block 5: A1_bottom * B_bottom_mid (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block([2, 3, 6, 7], [12, 13, 17, 18], [6, 9, 7, 10], U_blk, V_blk, W_blk)
    
    # Block 6: A1_bottom * B_bottom_right (2x2x1)
    U_blk, V_blk, W_blk = schoolbook_2x2x1()
    add_block([2, 3, 6, 7], [14, 19], [12, 13], U_blk, V_blk, W_blk)
    
    # Block 7: A2_top * B_top_left (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    add_block([8, 9], [0, 1, 5, 6], [2, 5], U_blk, V_blk, W_blk)
    
    # Block 8: A2_top * B_top_mid (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    add_block([8, 9], [2, 3, 7, 8], [8, 11], U_blk, V_blk, W_blk)
    
    # Block 9: A2_top * B_top_right (1x2x1)
    U_blk, V_blk, W_blk = schoolbook_1x2x1()
    add_block([8, 9], [4, 9], [14], U_blk, V_blk, W_blk)
    
    # Block 10: A2_bottom * B_bottom_left (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    add_block([10, 11], [10, 11, 15, 16], [2, 5], U_blk, V_blk, W_blk)
    
    # Block 11: A2_bottom * B_bottom_mid (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    add_block([10, 11], [12, 13, 17, 18], [8, 11], U_blk, V_blk, W_blk)
    
    # Block 12: A2_bottom * B_bottom_right (1x2x1)
    U_blk, V_blk, W_blk = schoolbook_1x2x1()
    add_block([10, 11], [14, 19], [14], U_blk, V_blk, W_blk)
    
    if not U_parts:
        return {
            "U": np.zeros((n*m, 1)).tolist(),
            "V": np.zeros((m*p, 1)).tolist(),
            "W": np.zeros((p*n, 1)).tolist()
        }
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
