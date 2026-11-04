import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen factors for 2x2x2
    # U_s, V_s, W_s are 4x7
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # Strassen decomposition for 2x2x2
    # M1 = (A10 + A11)(B00 + B01)
    # M2 = (A10 + A11)(B10 - B00)
    # M3 = (A00 - A10)(B00 + B10)
    # M4 = (A01 - A11)(B01 + B11)
    # M5 = A00 * B00
    # M6 = A11 * B11
    # M7 = (A00 + A11)(B10 - B01)
    
    # Indices for 2x2: i in {0,1}, j in {0,1}
    # U index: i*2 + j
    # V index: j*2 + k
    # W index: k*2 + i
    
    # M1
    U_s[0+0, 0] = 1 # A00
    U_s[0+1, 0] = 1 # A01
    U_s[1+0, 0] = 1 # A10
    U_s[1+1, 0] = 1 # A11
    V_s[0+0, 0] = 1 # B00
    V_s[0+1, 0] = 1 # B01
    V_s[1+0, 0] = 1 # B10
    V_s[1+1, 0] = 1 # B11
    W_s[0+0, 0] = 1 # C00
    W_s[0+1, 0] = 1 # C01
    W_s[1+0, 0] = 1 # C10
    W_s[1+1, 0] = 1 # C11
    
    # M2
    U_s[1+0, 1] = 1 # A10
    U_s[1+1, 1] = 1 # A11
    V_s[1+0, 1] = 1 # B10
    V_s[0+0, 1] = -1 # B00
    W_s[0+0, 1] = 1 # C00
    W_s[0+1, 1] = 1 # C01
    W_s[1+0, 1] = 1 # C10
    W_s[1+1, 1] = 1 # C11
    
    # M3
    U_s[0+0, 2] = 1 # A00
    U_s[1+0, 2] = -1 # A10
    V_s[0+0, 2] = 1 # B00
    V_s[1+0, 2] = 1 # B10
    W_s[0+0, 2] = 1 # C00
    W_s[0+1, 2] = 1 # C01
    W_s[1+0, 2] = 1 # C10
    W_s[1+1, 2] = 1 # C11
    
    # M4
    U_s[0+1, 3] = 1 # A01
    U_s[1+1, 3] = -1 # A11
    V_s[0+1, 3] = 1 # B01
    V_s[1+1, 3] = 1 # B11
    W_s[0+0, 3] = 1 # C00
    W_s[0+1, 3] = 1 # C01
    W_s[1+0, 3] = 1 # C10
    W_s[1+1, 3] = 1 # C11
    
    # M5
    U_s[0+0, 4] = 1 # A00
    V_s[0+0, 4] = 1 # B00
    W_s[0+0, 4] = 1 # C00
    
    # M6
    U_s[1+1, 5] = 1 # A11
    V_s[1+1, 5] = 1 # B11
    W_s[1+1, 5] = 1 # C11
    
    # M7
    U_s[0+0, 6] = 1 # A00
    U_s[1+1, 6] = 1 # A11
    V_s[1+0, 6] = 1 # B10
    V_s[0+1, 6] = -1 # B01
    W_s[0+1, 6] = 1 # C01
    W_s[1+0, 6] = 1 # C10
    
    # Factors for 2x2x1 (dot product)
    # U_1, V_1, W_1 are 4x4, 2x4, 2x4
    U_1 = np.zeros((4, 4))
    V_1 = np.zeros((2, 4))
    W_1 = np.zeros((2, 4))
    for i in range(2):
        for j in range(2):
            c = i * 2 + j
            U_1[i * 2 + j, c] = 1
            V_1[j, c] = 1
            W_1[i, c] = 1
            
    # Construct 4x4x4 factors (Rank 49)
    # U_44 = U_s (x) U_s
    # Need to permute rows to match i*4+j ordering.
    # Kronecker product U_s (x) U_s has rows indexed by (i1, i2, j1, j2) -> i1*4 + i2*2 + j1*2 + j2?
    # No, np.kron(A, B) row index is i_A * rows_B + i_B.
    # U_s rows are i*2+j.
    # So row index of kron is (i1*2+j1)*4 + (i2*2+j2) = i1*8 + j1*4 + i2*2 + j2.
    # We want row index i*4+j where i = i1*2+i2, j = j1*2+j2.
    # So target index = (i1*2+i2)*4 + (j1*2+j2) = i1*8 + i2*4 + j1*2 + j2.
    # We need a permutation matrix P such that P @ (U_s (x) U_s) has the correct row order.
    
    def permute_rows(mat, old_indices, new_indices):
        # old_indices: list of source row indices
        # new_indices: list of target row indices
        # Actually, just create a new matrix
        n_rows, n_cols = mat.shape
        new_mat = np.zeros((n_rows, n_cols))
        for old_idx, new_idx in zip(old_indices, new_indices):
            new_mat[new_idx] = mat[old_idx]
        return new_mat

    # Generate permutations for 4x4x4
    # Source indices from kron: i1*8 + j1*4 + i2*2 + j2
    # Target indices: i1*8 + i2*4 + j1*2 + j2
    src_idx_44 = []
    tgt_idx_44 = []
    for i1 in range(2):
        for j1 in range(2):
            for i2 in range(2):
                for j2 in range(2):
                    src = i1 * 8 + j1 * 4 + i2 * 2 + j2
                    tgt = i1 * 8 + i2 * 4 + j1 * 2 + j2
                    src_idx_44.append(src)
                    tgt_idx_44.append(tgt)
    
    U_44_raw = np.kron(U_s, U_s)
    V_44_raw = np.kron(V_s, V_s)
    W_44_raw = np.kron(W_s, W_s)
    
    # The permutation is the same for U, V, W?
    # U index: i*m+j. For 4x4, i in 0..3, j in 0..3.
    # V index: j*p+k. For 4x4, j in 0..3, k in 0..3.
    # W index: k*n+i. For 4x4, k in 0..3, i in 0..3.
    # The structure is symmetric for U and V and W in terms of index composition.
    # So the same permutation applies.
    
    U_44 = permute_rows(U_44_raw, src_idx_44, tgt_idx_44)
    V_44 = permute_rows(V_44_raw, src_idx_44, tgt_idx_44)
    W_44 = permute_rows(W_44_raw, src_idx_44, tgt_idx_44)
    
    # Construct 4x4x2 factors (Rank 28)
    # U_42 = U_s (x) U_1
    # U_s rows: i1*2+j1. U_1 rows: i2*2+j2.
    # Kron rows: (i1*2+j1)*4 + (i2*2+j2) = i1*8 + j1*4 + i2*2 + j2.
    # Target U index: i*4+j where i=i1*2+i2, j=j1*2+j2.
    # Tgt: i1*8 + i2*4 + j1*2 + j2.
    # Same permutation as above?
    # Yes, the indices for i and j are structured the same way.
    
    U_42_raw = np.kron(U_s, U_1)
    # V_42 = V_s (x) V_1
    # V_s rows: j1*2+k1. V_1 rows: j2*1+k2? No, V_1 is 2x4. Rows j2 in 0..1.
    # Wait, V_1 shape is (2, 4). Rows are j.
    # Kron rows: (j1*2+k1)*2 + j2 = j1*4 + k1*2 + j2.
    # Target V index for 4x2 part: j*p+k. Here p=2 (local).
    # j = j1*2+j2, k = k1 (local k for 4x2? No, k is 0..1).
    # Wait, in 4x4x2, B is 4x2. j in 0..3, k in 0..1.
    # j = j1*2+j2. k = k1.
    # Tgt index: j*2 + k = (j1*2+j2)*2 + k1 = j1*4 + j2*2 + k1.
    # Src index: j1*4 + k1*2 + j2.
    # We need to permute V_42 rows.
    
    src_idx_V42 = []
    tgt_idx_V42 = []
    for j1 in range(2):
        for k1 in range(2):
            for j2 in range(2):
                # k2 is always 0 for V_1? No, V_1 has 1 column?
                # V_1 is 2x4. The 4 columns correspond to the 4 terms.
                # The row index of V_1 is j2 (0 or 1).
                # So src row index is (j1*2+k1)*2 + j2.
                src = j1 * 4 + k1 * 2 + j2
                tgt = j1 * 4 + j2 * 2 + k1
                src_idx_V42.append(src)
                tgt_idx_V42.append(tgt)
                
    V_42_raw = np.kron(V_s, V_1)
    V_42 = permute_rows(V_42_raw, src_idx_V42, tgt_idx_V42)
    
    # W_42 = W_s (x) W_1
    # W_s rows: k1*2+i1. W_1 rows: i2.
    # Kron rows: (k1*2+i1)*2 + i2 = k1*4 + i1*2 + i2.
    # Target W index for 4x2 part: k*n+i. Here n=4.
    # k = k1 (local 0 or 1). i = i1*2+i2.
    # Tgt index: k*4 + i = k1*4 + (i1*2+i2) = k1*4 + i1*2 + i2.
    # Src index is the same! No permutation needed for W_42?
    # Let's check.
    # Src: k1*4 + i1*2 + i2.
    # Tgt: k1*4 + i1*2 + i2.
    # Yes, identical.
    
    W_42_raw = np.kron(W_s, W_1)
    W_42 = W_42_raw
    
    # Assemble global matrices
    rank = 49 + 28
    U = np.zeros((16, rank))
    V = np.zeros((24, rank))
    W = np.zeros((24, rank))
    
    # Part 1: 4x4x4 (cols 0-48)
    U[:, 0:49] = U_44
    V[0:16, 0:49] = V_44
    W[0:16, 0:49] = W_44
    
    # Part 2: 4x4x2 (cols 49-76)
    U[:, 49:77] = U_42
    # V rows 16-23 correspond to j*6 + k where k in 4,5.
    # V_42 rows are j*2 + k_local where k_local in 0,1.
    # j is 0..3.
    for j in range(4):
        for k_local in range(2):
            global_k = 4 + k_local
            global_row = j * 6 + global_k
            local_row = j * 2 + k_local
            V[global_row, 49:77] = V_42[local_row, :]
            
    # W rows 16-23 correspond to k*4 + i where k in 4,5.
    # W_42 rows are k_local*4 + i where k_local in 0,1.
    for k_local in range(2):
        global_k = 4 + k_local
        for i in range(4):
            global_row = global_k * 4 + i
            local_row = k_local * 4 + i
            W[global_row, 49:77] = W_42[local_row, :]
            
    # Convert to list of lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
