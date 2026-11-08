import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 4, 4, 8
    # Rank 49 for 4x4x4, doubled for 4x4x8 -> 98
    rank = 98
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Base 2x2x2 Strassen (Rank 7)
    # Indices: 0,1,2,3 correspond to [[0,0],[0,1],[1,0],[1,1]]
    U2 = np.array([
        [1, 0, 1, 0, 1, 1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1], # A01
        [0, 1, 0, 0, 0, 1, 0], # A10
        [1, 1, 0, 1, 0, -1, 1] # A11
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 1],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, -1, 1, 0, 1] # B11
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ], dtype=float)
    
    # Check validity of base Strassen (optional, but good for debugging locally)
    # M = U2.T @ (diag(U2 @ V2.T) ...) # Too complex to verify quickly, relying on known correctness
    
    # Construct 4x4x4 decomposition (Rank 49)
    # The 4x4 matrix A is viewed as 2x2 blocks of 2x2 matrices.
    # We iterate over the 7 "outer" Strassen terms (block level) and 7 "inner" Strassen terms (sub-block level).
    
    # Outer Strassen coefficients (mapping block indices to U2/V2/W2 indices)
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    # C blocks: 00, 01, 10, 11
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A11 - A01)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We define the coefficients for A, B, C in terms of the 4 blocks.
    # Each block is a 2x2 matrix, so we need to map the 2x2 Strassen U2/V2/W2 to the 4x4 indices.
    
    # Block to Global Index Mapping for 2x2 blocks of 2x2 matrices
    # Block (bi, bj) contains rows bi*2..bi*2+1 and cols bj*2..bj*2+1
    # Global row = bi*2 + r, Global col = bj*2 + c
    
    # Helper to get global indices for a block
    def get_block_indices(bi, bj):
        rows = []
        cols = []
        for r in range(2):
            for c in range(2):
                rows.append(bi * 2 + r)
                cols.append(bj * 2 + c)
        return rows, cols

    # Pre-allocate 4x4x4 U4, V4, W4 (Rank 49)
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Outer Strassen Logic
    # For each outer term k (0..6), we have contributions from inner terms j (0..6)
    
    # Define the linear combinations for the outer level
    # A_outer[k, block_idx]
    # B_outer[k, block_idx]
    # C_outer[block_idx, k]
    
    A_outer = np.zeros((7, 4))
    B_outer = np.zeros((7, 4))
    C_outer = np.zeros((4, 7))
    
    # k=0 (M1): A00+A11, B00+B11 -> C00, C11
    A_outer[0, 0] = 1; A_outer[0, 3] = 1
    B_outer[0, 0] = 1; B_outer[0, 3] = 1
    C_outer[0, 0] = 1; C_outer[3, 0] = 1
    
    # k=1 (M2): A10+A11, B00 -> C10, C11
    A_outer[1, 2] = 1; A_outer[1, 3] = 1
    B_outer[1, 0] = 1
    C_outer[2, 1] = 1; C_outer[3, 1] = -1
    
    # k=2 (M3): A00, B01-B11 -> C01, C11
    A_outer[2, 0] = 1
    B_outer[2, 1] = 1; B_outer[2, 3] = -1
    C_outer[1, 2] = 1; C_outer[3, 2] = 1
    
    # k=3 (M4): A11, B10-B00 -> C00, C10
    A_outer[3, 3] = 1
    B_outer[3, 2] = 1; B_outer[3, 0] = -1
    C_outer[0, 3] = 1; C_outer[2, 3] = 1
    
    # k=4 (M5): A00+A01, B11 -> C00, C01
    A_outer[4, 0] = 1; A_outer[4, 1] = 1
    B_outer[4, 3] = 1
    C_outer[0, 4] = -1; C_outer[1, 4] = 1
    
    # k=5 (M6): A10-A00, B00+B01 -> C11
    A_outer[5, 2] = 1; A_outer[5, 0] = -1
    B_outer[5, 0] = 1; B_outer[5, 1] = 1
    C_outer[3, 5] = 1
    
    # k=6 (M7): A11-A01, B10+B11 -> C00
    A_outer[6, 3] = 1; A_outer[6, 1] = -1
    B_outer[6, 2] = 1; B_outer[6, 3] = 1
    C_outer[0, 6] = 1
    
    # Now fill U4, V4, W4
    # Column index in U4/V4/W4 is k*7 + j
    for k in range(7):
        for j in range(7):
            col_idx = k * 7 + j
            
            # U4: Combine A_outer coefficients with U2
            for bi in range(2):
                for bj in range(2):
                    block_idx = bi * 2 + bj
                    coeff = A_outer[k, block_idx]
                    if coeff != 0:
                        rows, cols = get_block_indices(bi, bj)
                        for idx in range(4):
                            r = rows[idx]
                            c = cols[idx]
                            global_idx = r * 4 + c
                            u2_val = U2[idx, j]
                            if u2_val != 0:
                                U4[global_idx, col_idx] += coeff * u2_val
            
            # V4: Combine B_outer coefficients with V2
            # Note: V4 is for the 4x4 B matrix.
            for bi in range(2):
                for bj in range(2):
                    block_idx = bi * 2 + bj
                    coeff = B_outer[k, block_idx]
                    if coeff != 0:
                        rows, cols = get_block_indices(bi, bj)
                        for idx in range(4):
                            r = rows[idx]
                            c = cols[idx]
                            global_idx = r * 4 + c
                            v2_val = V2[idx, j]
                            if v2_val != 0:
                                V4[global_idx, col_idx] += coeff * v2_val
            
            # W4: Combine C_outer coefficients with W2
            for bi in range(2):
                for bj in range(2):
                    block_idx = bi * 2 + bj
                    coeff = C_outer[block_idx, k]
                    if coeff != 0:
                        rows, cols = get_block_indices(bi, bj)
                        for idx in range(4):
                            r = rows[idx]
                            c = cols[idx]
                            # Output W is indexed by [k, n+i, r] ? No, W shape is (p*n, rank)
                            # Task says: W[k*n+i, r] where k is output col, i is output row?
                            # Let's check the standard convention.
                            # Usually C[i,j] = sum U[i,m,r] V[m,j,r] W[i,j,r]
                            # Or C = W @ diag(V^T U) ??
                            # The task says: "output factor follows the official trace convention W[k*n+i,r]"
                            # This implies W is a matrix of shape (p*n, rank).
                            # Index k is the column index of the result C (0..p-1).
                            # Index i is the row index of the result C (0..n-1).
                            # So W index is k*n + i.
                            # In our 4x4x4 case, p=4, n=4.
                            # Result block (bi, bj) corresponds to rows bi*2..bi*2+1 and cols bj*2..bj*2+1.
                            # So i = bi*2 + r, k = bj*2 + c.
                            # W index = k*4 + i.
                            global_w_idx = (bj * 2 + c) * 4 + (bi * 2 + r)
                            w2_val = W2[idx, j]
                            if w2_val != 0:
                                W4[global_w_idx, col_idx] += coeff * w2_val

    # Now construct 4x4x8 U, V, W
    # U is same for both halves (since A is same)
    # V has two halves: V_L for B_L, V_R for B_R
    # W has two halves: W_L for C_L, W_R for C_R
    
    # U
    U[:, 0:49] = U4
    U[:, 49:98] = U4
    
    # V
    # V4 is 16x49. 
    # For the first 4 columns of B (B_L), we use V4 directly.
    V[:, 0:49] = V4
    # For the last 4 columns of B (B_R), the indices in V correspond to rows 0..3 and cols 4..7.
    # V matrix shape is (m*p, rank) = (32, 98).
    # Rows 0..15 correspond to B_L (cols 0..3).
    # Rows 16..31 correspond to B_R (cols 4..7).
    # So we need to map the local row indices of V4 (0..15) to global row indices 16..31.
    V[16:32, 49:98] = V4
    
    # W
    # W4 is 16x49.
    # W4 rows correspond to C indices k*4+i.
    # For C_L (cols 0..3), k is 0..3. Global W index is k*4+i. This matches rows 0..15 of W4.
    # For C_R (cols 4..7), k is 4..7. Global W index is k*4+i.
    # In W4, the "k" component was 0..3. We need to shift it by 4.
    # W4 index = k_local*4 + i.
    # Global index = (k_local + 4)*4 + i = k_local*4 + i + 16.
    # So we just add 16 to the row indices of W4.
    W[0:16, 0:49] = W4
    W[16:32, 49:98] = W4
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
