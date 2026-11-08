import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # We will construct the decomposition block by block
    # Block 1: 2x3x2 (cols 0,1 of B)
    # Block 2: 2x3x2 (cols 2,3 of B)
    # Block 3: 2x3x3 (cols 4,5,6 of B)
    
    # Helper to add a decomposition
    # U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    # We will build lists of columns for U, V, W
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_decomp(U_block, V_block, W_block, col_offset_b=0, col_offset_c=0):
        """
        U_block: (n*m) x r
        V_block: (m*p_block) x r
        W_block: (p_block*n) x r
        """
        r = U_block.shape[1]
        # U is same for all blocks
        U_cols.append(U_block)
        
        # V needs to be embedded in (m*p) x r
        V_full = np.zeros((m*p, r))
        # V_block corresponds to columns col_offset_b to col_offset_b + p_block
        # V_block indices: j in 0..m-1, k in 0..p_block-1
        # V_full indices: j in 0..m-1, k in col_offset_b .. col_offset_b + p_block - 1
        for j in range(m):
            for k in range(p_block):
                V_full[j*p + col_offset_b + k, :] = V_block[j*p_block + k, :]
        V_cols.append(V_full)
        
        # W needs to be embedded in (p*n) x r
        W_full = np.zeros((p*n, r))
        # W_block indices: k in 0..p_block-1, i in 0..n-1
        # W_full indices: k in col_offset_c .. col_offset_c + p_block - 1, i in 0..n-1
        for k in range(p_block):
            for i in range(n):
                W_full[(col_offset_c + k)*n + i, :] = W_block[k*n + i, :]
        W_cols.append(W_full)

    # --- Block 1: 2x3x2 ---
    # A is 2x3, B is 3x2
    # Split A into A11 (2x2) and A12 (2x1)
    # Split B into B11 (2x2) and B12 (1x2)
    # C = A11 B11 + A12 B12
    
    # Strassen 2x2x2 for A11 B11
    # A11 indices: A[0,0], A[0,1], A[1,0], A[1,1] -> U indices 0,1,2,3
    # B11 indices: B[0,0], B[0,1], B[1,0], B[1,1] -> V indices 0,1,2,3 (local)
    # W indices: C[0,0], C[0,1], C[1,0], C[1,1] -> W indices 0,1,2,3 (local)
    
    U_s = np.zeros((6, 7))
    V_s = np.zeros((6, 7)) # m*p_block = 3*2 = 6
    W_s = np.zeros((4, 7)) # p_block*n = 2*2 = 4
    
    # Strassen factors
    # M1: (A00+A11)(B00+B11)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    # M2: (A10+A11)B00
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1
    # M3: A00(B01-B11)
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    W_s[1, 2] = 1
    # M4: A11(B10-B00)
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] = 1; W_s[2, 3] = 1
    # M5: (A00+A01)B11
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[1, 4] = 1
    # M6: (A10-A00)(B00+B01)
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[3, 5] = 1
    # M7: (A01-A11)(B10+B11)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1; W_s[1, 6] = -1
    
    # Schoolbook 2x1x2 for A12 B12
    # A12: A[0,2], A[1,2] -> U indices 4, 5
    # B12: B[2,0], B[2,1] -> V indices 4, 5 (local)
    # C: 2x2
    U_sb = np.zeros((6, 4))
    V_sb = np.zeros((6, 4))
    W_sb = np.zeros((4, 4))
    
    # Term 1: A[0,2] B[2,0] -> C[0,0]
    U_sb[4, 0] = 1
    V_sb[4, 0] = 1
    W_sb[0, 0] = 1
    # Term 2: A[0,2] B[2,1] -> C[0,1]
    U_sb[4, 1] = 1
    V_sb[5, 1] = 1
    W_sb[1, 1] = 1
    # Term 3: A[1,2] B[2,0] -> C[1,0]
    U_sb[5, 2] = 1
    V_sb[4, 2] = 1
    W_sb[2, 2] = 1
    # Term 4: A[1,2] B[2,1] -> C[1,1]
    U_sb[5, 3] = 1
    V_sb[5, 3] = 1
    W_sb[3, 3] = 1
    
    # Combine Block 1
    U_b1 = np.hstack([U_s, U_sb])
    V_b1 = np.hstack([V_s, V_sb])
    W_b1 = np.hstack([W_s, W_sb])
    
    add_decomp(U_b1, V_b1, W_b1, col_offset_b=0, col_offset_c=0)
    
    # --- Block 2: 2x3x2 (cols 2,3) ---
    # Same structure, just shifted
    U_b2 = U_b1
    V_b2 = V_b1
    W_b2 = W_b1
    add_decomp(U_b2, V_b2, W_b2, col_offset_b=2, col_offset_c=2)
    
    # --- Block 3: 2x3x3 (cols 4,5,6) ---
    # A is 2x3, B is 3x3
    # Split A into A11 (2x2) and A12 (2x1)
    # Split B into B11 (2x3) and B12 (1x3)
    # C = A11 B11 + A12 B12
    
    # A11 B11 is 2x2 by 2x3. Rank 11.
    # B11 is 2x3. Split into B11a (2x2) and B11b (2x1).
    # A11 B11a: Rank 7 (Strassen)
    # A11 B11b: Rank 4 (Schoolbook 2x2x1)
    
    U_223 = np.zeros((6, 11))
    V_223 = np.zeros((6, 11)) # m*p_block = 3*3 = 9? No, B11 is 2x3. V indices for B11 are 0..5 (2x3=6 entries).
    # Wait, V_block for 2x3x3 has shape (m*p_block) = 3*3 = 9.
    # B11 uses rows 0,1 of B. So V indices 0..5.
    # B12 uses row 2 of B. So V indices 6..8.
    
    # Strassen for A11 (2x2) and B11a (2x2)
    # B11a cols 0,1 of B11. Global cols 4,5.
    # In local V_223 (size 6x11), B11a indices are 0,1,2,3.
    U_s3 = np.zeros((6, 7))
    V_s3 = np.zeros((6, 7))
    W_s3 = np.zeros((6, 7)) # p_block*n = 3*2 = 6. C is 2x3.
    # W indices: C[0,0], C[0,1], C[1,0], C[1,1] -> 0,1,2,3.
    # C[0,2], C[1,2] are from B11b.
    
    # Copy Strassen U, V
    U_s3[:, :7] = U_s
    # V_s3: B11a is cols 0,1 of B11.
    V_s3[0, 0] = 1; V_s3[3, 0] = 1
    V_s3[0, 1] = 1
    V_s3[1, 2] = 1; V_s3[3, 2] = -1
    V_s3[2, 3] = 1; V_s3[0, 3] = -1
    V_s3[3, 4] = 1
    V_s3[0, 5] = 1; V_s3[1, 5] = 1
    V_s3[2, 6] = 1; V_s3[3, 6] = 1
    
    # W_s3: Same as W_s but embedded in 2x3
    W_s3[0, 0] = 1; W_s3[3, 0] = 1
    W_s3[2, 1] = 1
    W_s3[1, 2] = 1
    W_s3[0, 3] = 1; W_s3[2, 3] = 1
    W_s3[0, 4] = -1; W_s3[1, 4] = 1
    W_s3[3, 5] = 1
    W_s3[0, 6] = 1; W_s3[1, 6] = -1
    
    # Schoolbook 2x2x1 for A11 and B11b (2x1)
    # B11b is col 2 of B11. Global col 6.
    # V index 5 (row 0, col 2 of B11 is index 0*2+2=2? No. B11 is 2x3.
    # B11[0,0], B11[0,1], B11[0,2] -> 0,1,2
    # B11[1,0], B11[1,1], B11[1,2] -> 3,4,5
    # So B11b is indices 2, 5.
    
    U_sb3 = np.zeros((6, 4))
    V_sb3 = np.zeros((6, 4))
    W_sb3 = np.zeros((6, 4))
    
    # Term 1: A[0,0] B[0,2] -> C[0,2]
    U_sb3[0, 0] = 1
    V_sb3[2, 0] = 1
    W_sb3[1, 0] = 1 # C[0,2] is index 0*3+2=2? No.
    # W index: k*n+i.
    # C is 2x3. i in 0..1, k in 0..2.
    # C[0,2] -> k=2, i=0 -> 2*2+0 = 4.
    # C[1,2] -> k=2, i=1 -> 2*2+1 = 5.
    W_sb3[4, 0] = 1
    
    # Term 2: A[0,1] B[0,2] -> C[0,2]
    U_sb3[1, 1] = 1
    V_sb3[2, 1] = 1
    W_sb3[4, 1] = 1
    
    # Term 3: A[1,0] B[1,2] -> C[1,2]
    U_sb3[2, 2] = 1
    V_sb3[5, 2] = 1
    W_sb3[5, 2] = 1
    
    # Term 4: A[1,1] B[1,2] -> C[1,2]
    U_sb3[3, 3] = 1
    V_sb3[5, 3] = 1
    W_sb3[5, 3] = 1
    
    U_223[:, :7] = U_s3
    U_223[:, 7:11] = U_sb3
    V_223[:, :7] = V_s3
    V_223[:, 7:11] = V_sb3
    W_223[:, :7] = W_s3
    W_223[:, 7:11] = W_sb3
    
    # Now A12 B12: 2x1 by 1x3. Rank 6.
    # A12: A[0,2], A[1,2] -> U indices 4,5.
    # B12: B[2,0], B[2,1], B[2,2] -> V indices 6,7,8 (in 3x3 block).
    # C: 2x3.
    
    U_113 = np.zeros((6, 6))
    V_113 = np.zeros((9, 6)) # m*p_block = 3*3 = 9
    W_113 = np.zeros((6, 6))
    
    # Term 1: A[0,2] B[2,0] -> C[0,0]
    U_113[4, 0] = 1
    V_113[6, 0] = 1
    W_113[0, 0] = 1 # C[0,0] -> k=0,i=0 -> 0
    
    # Term 2: A[0,2] B[2,1] -> C[0,1]
    U_113[4, 1] = 1
    V_113[7, 1] = 1
    W_113[1, 1] = 1 # C[0,1] -> k=0,i=0? No. k=1?
    # C[0,1] -> i=0, k=1. Index 1*2+0 = 2.
    W_113[2, 1] = 1
    
    # Term 3: A[0,2] B[2,2] -> C[0,2]
    U_113[4, 2] = 1
    V_113[8, 2] = 1
    W_113[4, 2] = 1 # C[0,2] -> i=0, k=2. Index 2*2+0 = 4.
    
    # Term 4: A[1,2] B[2,0] -> C[1,0]
    U_113[5, 3] = 1
    V_113[6, 3] = 1
    W_113[1, 3] = 1 # C[1,0] -> i=1, k=0. Index 0*2+1 = 1.
    
    # Term 5: A[1,2] B[2,1] -> C[1,1]
    U_113[5, 4] = 1
    V_113[7, 4] = 1
    W_113[3, 4] = 1 # C[1,1] -> i=1, k=1. Index 1*2+1 = 3.
    
    # Term 6: A[1,2] B[2,2] -> C[1,2]
    U_113[5, 5] = 1
    V_113[8, 5] = 1
    W_113[5, 5] = 1 # C[1,2] -> i=1, k=2. Index 2*2+1 = 5.
    
    # Combine Block 3
    U_b3 = np.hstack([U_223, U_113])
    V_b3 = np.hstack([V_223, V_113])
    W_b3 = np.hstack([W_223, W_113])
    
    add_decomp(U_b3, V_b3, W_b3, col_offset_b=4, col_offset_c=4)
    
    # Finalize
    U = np.hstack(U_cols)
    V = np.hstack(V_cols)
    W = np.hstack(W_cols)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print(res["U"][0])
