import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0

    # Schoolbook fallback
    rank_school = n * m * p
    U_sb = np.zeros((n*m, rank_school))
    V_sb = np.zeros((m*p, rank_school))
    W_sb = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1

    best_rank = rank_school
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    def _verify_decomposition(U, V, W, n, m, p):
        r = U.shape[1]
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        res = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        return np.allclose(res, 1.0, atol=1e-9)

    # Winograd 2x2 factors
    U2 = np.array([
        [1, 0, 1, 0, 0],
        [0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0]
    ], dtype=float)
    
    V2 = np.array([
        [1, 0, 0, 1, 0],
        [0, 1, 1, 0, -1],
        [1, 1, 0, 1, 0],
        [1, 0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, -1, 1, -1, 1],
        [1, -1, 1, 0, -1],
        [1, 1, -1, 1, 0],
        [1, 1, -1, 0, 1]
    ], dtype=float)

    # Construct 4x4x4 Winograd factors (Rank 25)
    # Kronecker product
    U4_kron = np.kron(U2, U2) # 16x25
    V4_kron = np.kron(V2, V2) # 16x25
    W4_kron = np.kron(W2, W2) # 16x25

    # Reshape to match indices
    # U4_kron rows are (i1, i2) where i1, i2 in 0..1.
    # We need rows indexed by i*4 + j.
    # The kron product order is (row1, row2).
    # If U2 is indexed by [0,1], U4_kron is indexed by [0,1]x[0,1].
    # This corresponds to i*2 + j?
    # Let's check the mapping.
    # A is 4x4. A[i,j].
    # Winograd 4x4 treats A as 2x2 blocks of 2x2.
    # The Kronecker product structure implies:
    # U4[i*4+j, t] = U2[i*2+u, t1] * U2[j*2+v, t2] where t = t1*5 + t2.
    
    # We need to permute the rows of U4_kron to match i*4+j order.
    # U4_kron row index is (i_high, i_low) in base 2?
    # np.kron(A, B) has row index i_A * n_B + i_B.
    # So row 0 is (0,0), row 1 is (0,1), row 2 is (1,0), row 3 is (1,1).
    # This corresponds to i=0,j=0; i=0,j=1; i=1,j=0; i=1,j=1.
    # This is exactly the i*2+j order for 2x2.
    # For 4x4, we need i*4+j.
    # The kron product gives (i_high, i_low) which is i_high*2 + i_low.
    # This is NOT i*4+j.
    # i*4+j is (i_high*2 + i_low) * 2 + j? No.
    # i is 0..3. j is 0..3.
    # i*4+j:
    # 0: 0*4+0 = 0 (i=0, j=0) -> (0,0)
    # 1: 0*4+1 = 1 (i=0, j=1) -> (0,1)
    # 2: 0*4+2 = 2 (i=0, j=2) -> (0,2) -> (0, 1, 0) in bits?
    # The kron product row index is (i1, j1) for the first matrix and (i2, j2) for the second?
    # No, U2 is for 2x2.
    # U4 = U2 \otimes U2.
    # Row index of U4 is (r1, r2) where r1, r2 are row indices of U2.
    # r1 corresponds to the "block row" index (0 or 1).
    # r2 corresponds to the "row within block" index (0 or 1).
    # So r1 = i // 2, r2 = i % 2.
    # Thus row index in kron is r1 * 2 + r2 = i.
    # Wait, U2 has 4 rows.
    # U2 rows are 0,1,2,3 corresponding to a00, a01, a10, a11.
    # So r1 is the index into {a00, a01, a10, a11}.
    # This means U2 already encodes the 2x2 structure.
    # So U4_kron row index is (r1, r2) where r1, r2 in 0..3.
    # This is NOT the 4x4 index i*4+j.
    
    # We need to map the 4x4 indices (i,j) to the Winograd structure.
    # The Winograd 4x4 algorithm uses the 2x2 factors on the 2x2 blocks.
    # Block (0,0) uses U2 rows for a00,a01,a10,a11 of the block.
    # This is getting complicated to map manually.
    
    # Simpler: Just use the schoolbook for the 5th column and Winograd for the rest?
    # No, let's stick to the Rank 29 construction but verify it.
    # If the mapping is wrong, _verify_decomposition will fail, and we return schoolbook.
    
    # Let's try to construct the Rank 29 directly.
    rank_candidate = 29
    U_cand = np.zeros((16, rank_candidate))
    V_cand = np.zeros((20, rank_candidate))
    W_cand = np.zeros((20, rank_candidate)) # p*n = 5*4 = 20

    # Part 1: Winograd for 4x4x4 (cols 0-3 of B, cols 0-3 of C)
    # We need to map U4_kron, V4_kron, W4_kron to the correct indices.
    # U4_kron is 16x25. Rows are (r1, r2) where r1,r2 in 0..3.
    # r1 corresponds to i_block * 2 + i_in_block?
    # Actually, for 4x4, the indices are i,j in 0..3.
    # The Kronecker product of two 2x2 tensors gives a 4x4x4 tensor.
    # The row index of U4_kron is (i1, i2) where i1, i2 are indices of U2.
    # i1 = i // 2, i2 = i % 2? No, U2 has 4 rows.
    # U2 row 0: a00. Row 1: a01. Row 2: a10. Row 3: a11.
    # So U2 row index is (row, col) of the 2x2 matrix.
    # So U4_kron row index is (r1_row, r1_col, r2_row, r2_col).
    # This corresponds to the 4x4 matrix index i = r1_row*2 + r2_row, j = r1_col*2 + r2_col.
    # So we need to permute the rows of U4_kron to match i*4+j.
    
    # Map kron_row to i*4+j
    # kron_row = r1*4 + r2 (since U2 has 4 rows)
    # r1 is (row1, col1), r2 is (row2, col2)
    # i = row1*2 + row2
    # j = col1*2 + col2
    # target_row = i*4 + j
    
    perm_U = np.zeros(16, dtype=int)
    for r1 in range(4):
        row1 = r1 // 2
        col1 = r1 % 2
        for r2 in range(4):
            row2 = r2 // 2
            col2 = r2 % 2
            kron_idx = r1 * 4 + r2
            i = row1 * 2 + row2
            j = col1 * 2 + col2
            target_idx = i * 4 + j
            perm_U[target_idx] = kron_idx

    # Apply permutation to U4_kron
    U_winograd = U4_kron[perm_U, :]
    
    # Similarly for V4_kron (B is 4x4 for this part)
    # V2 rows are b00, b01, b10, b11.
    # Same permutation.
    V_winograd = V4_kron[perm_U, :]
    
    # W4_kron (C is 4x4)
    # W2 rows are c00, c01, c10, c11.
    # Same permutation.
    W_winograd = W4_kron[perm_U, :]
    
    # Place into candidate matrices
    # U_cand[:, 0:25] = U_winograd
    # V_cand: B is 4x5. Winograd uses B[0:4, 0:4].
    # V_winograd rows are j*4+k (for k<4).
    # We need to place them in V_cand rows j*5+k.
    for j in range(4):
        for k in range(4):
            v_row = j * 4 + k
            target_v_row = j * 5 + k
            V_cand[target_v_row, 0:25] = V_winograd[v_row, :]
            
    # W_cand: C is 4x5. Winograd uses C[0:4, 0:4].
    # W_winograd rows are k*4+i (for k<4).
    # We need to place them in W_cand rows k*5+i? No, W is p*n.
    # W index is k*n + i. Here n=4, p=5.
    # So W index is k*4 + i.
    # Winograd W_winograd rows are k*4 + i.
    # So we can just copy.
    W_cand[0:16, 0:25] = W_winograd
    U_cand[:, 0:25] = U_winograd

    # Part 2: 5th column (k=4)
    # C[:, 4] = A @ B[:, 4]
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # Add 4 terms for j=0..3? No, add 4 terms for the rank 4 decomposition of the 4x4x1 tensor.
    # We established rank 4 for the 5th column.
    # Terms t=0..3.
    # U[i*4+j, 25+t] = 1 if i==t else 0?
    # Let's use the rank 4 decomposition:
    # T[i,j,4] = 1 = sum_t u[i,t] v[j,t] w[4,t]
    # u[i,t] = delta(i,t)
    # v[j,t] = 1
    # w[4,t] = 1
    
    for t in range(4):
        col_idx = 25 + t
        # U: rows i*4+j. We want u[i,t] = 1 if i==t.
        # So for i=t, all j, U[t*4+j, col_idx] = 1.
        for j in range(4):
            U_cand[t*4+j, col_idx] = 1.0
            
        # V: rows j*5+k. We want v[j,t] = 1 for k=4.
        # So for all j, V[j*5+4, col_idx] = 1.
        for j in range(4):
            V_cand[j*5+4, col_idx] = 1.0
            
        # W: rows k*4+i. We want w[4,t] = 1 for all i.
        # So for k=4, all i, W[4*4+i, col_idx] = 1.
        for i in range(4):
            W_cand[4*4+i, col_idx] = 1.0

    # Verify
    if _verify_decomposition(U_cand, V_cand, W_cand, n, m, p):
        best_rank = 29
        best_U = U_cand
        best_V = V_cand
        best_W = W_cand

    def to_list(mat):
        return [[float(val) for val in row] for row in mat]

    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
