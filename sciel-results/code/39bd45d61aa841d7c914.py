import time
import numpy as np

DIMENSIONS = [4, 4, 4]
_DEADLINE = 125

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS

    # Strassen 2x2 decomposition matrices
    # Indices: 0:00, 1:01, 2:10, 3:11
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # 00
        [0, 0, 0, 0, 1, 0, 1],   # 01
        [0, 1, 0, 0, 0, 1, 0],   # 10
        [1, 1, 0, 1, 0, 0, -1]   # 11
    ])
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # 00
        [0, 0, 1, 0, 0, 1, 0],   # 01
        [0, 0, 0, 1, 0, 0, 1],   # 10
        [1, 0, -1, 0, 1, 0, 1]   # 11
    ])
    
    # W2 rows ordered by k*2+i (col*2+row):
    # 0: C_00, 1: C_10, 2: C_01, 3: C_11
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # 0: C_00
        [0, 1, 0, 1, 0, 0, 0],   # 1: C_10
        [0, 0, 1, 0, 1, 0, 0],   # 2: C_01
        [1, -1, 1, 0, 0, 1, 0]   # 3: C_11
    ])

    # Construct 4x4 decomposition using Kronecker products
    # The 4x4 matrix multiplication tensor is the tensor product of two 2x2 tensors.
    # U_4x4 = U2 (x) U2, V_4x4 = V2 (x) V2, W_4x4 = W2 (x) W2
    # Note: The indexing convention must be handled correctly.
    # U index: (i, j) -> i*4+j. In Kronecker, row index is (I,i) where I is block row, i is local row.
    # The order of Kronecker product np.kron(A, B) for matrices A (m1 x n1) and B (m2 x n2)
    # results in a matrix of size (m1*m2) x (n1*n2).
    # Row (I, i) and Col (R, r) is A[I, R] * B[i, r].
    # This matches the recursive structure where the "outer" block index varies slower.
    
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # The shapes are:
    # U4: (4*4) x (7*7) = 16 x 49
    # V4: (4*4) x (7*7) = 16 x 49
    # W4: (4*4) x (7*7) = 16 x 49
    
    # We need to ensure the row/col ordering matches the task's expected index conventions.
    # Task: U shape (n*m, rank) = (16, rank). Index i*m + j.
    # Our U4 row index from kron is I*4 + i? 
    # Let's check np.kron(A, B) row indexing.
    # If A is 4x7 and B is 4x7.
    # Row index in A: 0..3 (representing 00, 01, 10, 11).
    # Row index in B: 0..3.
    # Kronecker row: A_row * 4 + B_row.
    # A_row is the "outer" block index (I*2+J for U? No, U2 rows are 00, 01, 10, 11).
    # So A_row encodes (I, J) where I is block row, J is block col?
    # U2[0] is for block 00 (I=0, J=0).
    # U2[1] is for block 01 (I=0, J=1).
    # U2[2] is for block 10 (I=1, J=0).
    # U2[3] is for block 11 (I=1, J=1).
    # So U2 row index k corresponds to I = k//2, J = k%2.
    # Similarly for V2 and W2.
    
    # In np.kron(U2, U2), the row index is r_outer * 4 + r_inner.
    # r_outer corresponds to U2 row index k_out = I*2+J_out? No, U2 row index is just the block index.
    # So r_outer = I*2+J_outer? Wait.
    # The 4x4 matrix is divided into 2x2 blocks.
    # Block (I, J) is a 2x2 matrix.
    # The element at (i, j) in the 4x4 matrix is in block (I=i//2, J=j//2) at local pos (i%2, j%2).
    # U factor index is i*m + j = i*4 + j.
    # Let i = I*2 + i_loc, j = J*2 + j_loc.
    # Index = (I*2 + i_loc)*4 + (J*2 + j_loc) = I*8 + i_loc*4 + J*2 + j_loc.
    
    # Kronecker row index from np.kron(A, B):
    # A is 4x7 (rows 0..3), B is 4x7 (rows 0..3).
    # Row r = r_A * 4 + r_B.
    # r_A is the index in U2. U2 index 0 is block 00 (I=0, J=0).
    # U2 index 1 is block 01 (I=0, J=1).
    # U2 index 2 is block 10 (I=1, J=0).
    # U2 index 3 is block 11 (I=1, J=1).
    # So r_A = I*2 + J_outer? No, U2 doesn't have "outer". U2 is the decomposition for ONE 2x2 block multiplication?
    # No, Strassen decomposes C = AB where A,B are 2x2.
    # A is composed of 4 blocks? No, A is 2x2.
    # The recursive step: To multiply 4x4, we view it as 2x2 blocks of 2x2.
    # Let A = [[A00, A01], [A10, A11]] where Aij are 2x2.
    # The multiplication of the 2x2 block structure uses Strassen's 7 products.
    # Each product is a 2x2 matrix multiplication.
    # So for each of the 7 Strassen products, we perform a 2x2 matrix multiplication.
    # This means we apply the 2x2 Strassen decomposition to each of the 7 products.
    # Total rank = 7 * 7 = 49.
    
    # The U matrix for the 4x4 case:
    # The "outer" Strassen step determines WHICH 2x2 blocks of A and B are combined.
    # The "inner" Strassen step determines HOW those 2x2 blocks are multiplied.
    
    # Let's trace indices.
    # Outer step:
    # We compute M1 = (A00+A11)(B00+B11), etc.
    # The coefficient for A_ij in M_k is given by U2[k, ij].
    # So for a specific outer rank index k_out (0..6), the contribution to A_ij is U2[k_out, idx_ij].
    # Here idx_ij maps (I,J) to 0..3.
    
    # Inner step:
    # The product is C_k = M_k. M_k is a 2x2 matrix.
    # Wait, no. M_k is a 2x2 matrix. The multiplication A_block * B_block is a 2x2 * 2x2.
    # So we use the 2x2 Strassen decomposition for each M_k.
    # The inner rank index k_in (0..6).
    # The contribution of the inner multiplication to the local elements of A_block is U2[k_in, idx_ii].
    
    # Total rank index R = k_out * 7 + k_in.
    # U_4x4 [ (I*2+i)*4 + (J*2+j), R ] = U2 [ k_out, I*2+J ] * U2 [ k_in, i*2+j ]?
    # Let's check the row index of U2.
    # U2 rows are indexed by the OUTPUT block position? Or INPUT?
    # In the seed code:
    # U2[0] is for output 00?
    # U2[0] = [1, 0, 1, 0, 1, -1, 0]
    # Strassen U matrix usually maps Input A to Products.
    # C = A B.
    # U, V, W such that C_ij = sum_k U_ik V_jk W_k? No.
    # Tensor T_ijk. U is A's factor, V is B's factor, W is C's factor.
    # A is n x m. B is m x p. C is n x p.
    # U shape (n*m, rank). V shape (m*p, rank). W shape (n*p, rank).
    # For 2x2: n=2, m=2, p=2.
    # U shape (4, 7). Rows indexed by i*2+j.
    # V shape (4, 7). Rows indexed by j*2+k (B is 2x2, so m=2, p=2. Index j*p+k? B[j,k]. j is row of B, k is col of B. In A*B, A[i,j] B[j,k]. So j is the shared index.
    # V row index corresponds to B's element. B is 2x2. Elements B_00, B_01, B_10, B_11.
    # Index 0: 00, 1: 01, 2: 10, 3: 11.
    # W shape (4, 7). Rows indexed by i*2+k (C's element).
    
    # In the seed code:
    # U2 rows:
    # 0: 00 (i=0, j=0)
    # 1: 01 (i=0, j=1)
    # 2: 10 (i=1, j=0)
    # 3: 11 (i=1, j=1)
    
    # V2 rows:
    # 0: 00 (j=0, k=0) -> B_00
    # 1: 01 (j=0, k=1) -> B_01
    # 2: 10 (j=1, k=0) -> B_10
    # 3: 11 (j=1, k=1) -> B_11
    
    # W2 rows:
    # 0: C_00 (i=0, k=0)
    # 1: C_10 (i=1, k=0) ??
    # Seed code comment: "W2 rows ordered by k*2+i (col*2+row)".
    # 0: C_00 (k=0, i=0)
    # 1: C_10 (k=0, i=1) -> i=1, k=0. C[1,0].
    # 2: C_01 (k=1, i=0) -> i=0, k=1. C[0,1].
    # 3: C_11 (k=1, i=1) -> i=1, k=1. C[1,1].
    # So W2 index is k*2 + i.
    
    # Now for 4x4.
    # A is 4x4. Blocks A_IJ.
    # U_4x4 row index: i*4 + j.
    # i = I*2 + i_loc. j = J*2 + j_loc.
    # U_4x4[i*4+j, R] = U_outer[I*2+J, k_out] * U_inner[i_loc*2+j_loc, k_in]?
    # Let's verify.
    # The term is U[ (I*2+i_loc)*4 + (J*2+j_loc), k_out*7 + k_in ].
    # This equals U2[ I*2+J, k_out ] * U2[ i_loc*2+j_loc, k_in ].
    
    # This is exactly np.kron(U2, U2) if:
    # Row 1 of kron: I*2+J (outer) and i_loc*2+j_loc (inner).
    # np.kron(U2, U2) row index r = r_out * 4 + r_in.
    # r_out is index in U2 (0..3). r_in is index in U2 (0..3).
    # So r_out = I*2+J. r_in = i_loc*2+j_loc.
    # Row index in 4x4 U: (I*2+i_loc)*4 + (J*2+j_loc) = I*8 + i_loc*4 + J*2 + j_loc.
    # Kron row index: (I*2+J)*4 + (i_loc*2+j_loc) = I*8 + J*4 + i_loc*2 + j_loc.
    # These are DIFFERENT.
    # Seed code index: i*m + j = i*4 + j.
    # Kron index: (I*2+J)*4 + (i_loc*2+j_loc).
    # So we need to permute the rows of U4, V4, W4 to match the i*m+j, j*p+k, k*n+i conventions.
    
    # Let's create the permutation for U.
    # Target row index: idx_t = i*4 + j.
    # Kron row index: idx_k = (I*2+J)*4 + (i_loc*2+j_loc).
    # We need to map idx_k -> idx_t.
    # i = I*2 + i_loc. j = J*2 + j_loc.
    # idx_t = (I*2+i_loc)*4 + (J*2+j_loc) = 8*I + 4*i_loc + 2*J + j_loc.
    
    # We can just build the matrices using loops to be safe and clear, avoiding permutation errors.
    # But loops are what the previous code did.
    # To be distinct, I will use the Kron product and then apply a permutation matrix.
    
    # Permutation for U (16 rows)
    perm_U = np.zeros(16, dtype=int)
    for I in range(2):
        for J in range(2):
            for i_loc in range(2):
                for j_loc in range(2):
                    i = I*2 + i_loc
                    j = J*2 + j_loc
                    target_idx = i*4 + j
                    kron_idx = (I*2+J)*4 + (i_loc*2+j_loc)
                    perm_U[kron_idx] = target_idx
    # perm_U[kron_idx] = target_idx means U4_permuted[target_idx] = U4[kron_idx]?
    # Or U4_new[perm_U[i]] = U4[i]?
    # Let's use: U_final = U4[perm_U] if perm_U is the indices to pick.
    # We want U_final[row] where row is target_idx.
    # So we need an array P such that P[target_idx] = kron_idx.
    P_U = np.zeros(16, dtype=int)
    for I in range(2):
        for J in range(2):
            for i_loc in range(2):
                for j_loc in range(2):
                    i = I*2 + i_loc
                    j = J*2 + j_loc
                    target_idx = i*4 + j
                    kron_idx = (I*2+J)*4 + (i_loc*2+j_loc)
                    P_U[target_idx] = kron_idx
    U_final = U4[P_U]
    
    # Permutation for V (16 rows)
    # V row index: j*p + k = j*4 + k.
    # j = J*2 + j_loc? No, V is for B.
    # B is 4x4. Blocks B_JK.
    # B element B[j, k]. j is row, k is col.
    # j = J*2 + j_loc. k = K*2 + k_loc.
    # Target idx: j*4 + k = (J*2+j_loc)*4 + (K*2+k_loc) = 8*J + 4*j_loc + 2*K + k_loc.
    # Kron V4 is V2 (x) V2.
    # V2 rows are indexed by (J_block, K_block) -> 0..3.
    # V2 row index for block B_JK is J*2+K.
    # Inner V2 row index for local element b_jk is j_loc*2+k_loc.
    # Kron row: (J*2+K)*4 + (j_loc*2+k_loc) = 8*J + 4*K + 2*j_loc + k_loc.
    # Target: 8*J + 4*j_loc + 2*K + k_loc.
    
    P_V = np.zeros(16, dtype=int)
    for J in range(2):
        for K in range(2):
            for j_loc in range(2):
                for k_loc in range(2):
                    j = J*2 + j_loc
                    k = K*2 + k_loc
                    target_idx = j*4 + k
                    kron_idx = (J*2+K)*4 + (j_loc*2+k_loc)
                    P_V[target_idx] = kron_idx
    V_final = V4[P_V]
    
    # Permutation for W (16 rows)
    # W row index: k*n + i = k*4 + i. (Task convention: W[k*n+i, r])
    # C element C[i, k].
    # i = I*2 + i_loc. k = K*2 + k_loc.
    # Target idx: k*4 + i = (K*2+k_loc)*4 + (I*2+i_loc) = 8*K + 4*k_loc + 2*I + i_loc.
    # Kron W4 is W2 (x) W2.
    # W2 rows are indexed by (K_block, I_block) -> 0..3.
    # Recall W2 order: 0: C_00 (k=0,i=0), 1: C_10 (k=0,i=1), 2: C_01 (k=1,i=0), 3: C_11 (k=1,i=1).
    # So W2 index is k*2 + i.
    # Block index in W2 for C_IK is K*2 + I.
    # Inner W2 index for local c_ik is k_loc*2 + i_loc.
    # Kron row: (K*2+I)*4 + (k_loc*2+i_loc) = 8*K + 4*I + 2*k_loc + i_loc.
    # Target: 8*K + 4*k_loc + 2*I + i_loc.
    
    P_W = np.zeros(16, dtype=int)
    for I in range(2):
        for K in range(2):
            for i_loc in range(2):
                for k_loc in range(2):
                    i = I*2 + i_loc
                    k = K*2 + k_loc
                    target_idx = k*4 + i
                    kron_idx = (K*2+I)*4 + (k_loc*2+i_loc)
                    P_W[target_idx] = kron_idx
    W_final = W4[P_W]
    
    # Convert to list of lists for output
    U = U_final.tolist()
    V = V_final.tolist()
    W = W_final.tolist()
    
    # Pruning
    rank = 49
    keep = []
    for c in range(rank):
        if not (np.all(U_final[:, c] == 0) or np.all(V_final[:, c] == 0) or np.all(W_final[:, c] == 0)):
            keep.append(c)
            
    U = [[U[i][c] for c in keep] for i in range(16)]
    V = [[V[i][c] for c in keep] for i in range(16)]
    W = [[W[i][c] for c in keep] for i in range(16)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
