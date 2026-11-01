import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = 2, 4, 5
    dim_u = n * m  # 8
    dim_v = m * p  # 20
    dim_w = p * n  # 10
    
    rank = 36
    U = np.zeros((dim_u, rank), dtype=np.int64)
    V = np.zeros((dim_v, rank), dtype=np.int64)
    W = np.zeros((dim_w, rank), dtype=np.int64)
    
    # Part 1: 2x4 * 4x4 = 2x4, using Strassen with 2x2 outer blocks
    # A blocks (1x2 each):
    #   A11 = A[0, 0:2], A12 = A[0, 2:4]
    #   A21 = A[1, 0:2], A22 = A[1, 2:4]
    # B blocks (2x2 each):
    #   B11 = B[0:2, 0:2], B12 = B[0:2, 2:4]
    #   B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    # C blocks (1x2 each):
    #   C11 = C[0, 0:2], C12 = C[0, 2:4]
    #   C21 = C[1, 0:2], C22 = C[1, 2:4]
    
    # Inner multiplication: (1x2) * (2x2) -> (1x2), rank 4
    # For inner vectors u (1x2), matrix v (2x2), output w (1x2):
    #   w[0] = u[0]*v[0][0] + u[1]*v[1][0]
    #   w[1] = u[0]*v[0][1] + u[1]*v[1][1]
    # Rank-1 terms: (u[i], v[i][j], w[j]) for i in {0,1}, j in {0,1}
    
    # Strassen A coefficients for blocks (A11, A12, A21, A22)
    # Each is a 1x2 block, so the "value" at position (row, col) within the block
    # A_strassen[r][block] = coefficient for that A block in term r
    A_strassen = [
        [1, 0, 0, 1],   # M1: A11 + A22
        [0, 0, 1, 1],   # M2: A21 + A22
        [1, 0, 0, 0],   # M3: A11
        [0, 0, 0, 1],   # M4: A22
        [1, 1, 0, 0],   # M5: A11 + A12
        [-1, 0, 1, 0],  # M6: A21 - A11
        [0, 1, 0, -1]   # M7: A12 - A22
    ]
    
    # Strassen B coefficients for blocks (B11, B12, B21, B22)
    # Each is a 2x2 block
    B_strassen = [
        [[1, 0], [0, 1]],   # M1: B11 + B22
        [[1, 0], [0, 0]],   # M2: B11
        [[0, 0], [1, -1]],  # M3: B21 - B22
        [[-1, 1], [0, 0]],  # M4: B12 - B11
        [[0, 0], [0, 1]],   # M5: B22
        [[1, 1], [0, 0]],   # M6: B11 + B12
        [[0, 0], [1, 1]]    # M7: B21 + B22
    ]
    
    # C coefficients: which C blocks receive each Strassen term's contribution
    # C11 = M1 - M4 - M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6 - M7
    C_strassen = [
        [1, 0, 0, 0],   # M1: C11
        [0, 0, 1, 0],   # M2: C21
        [0, 1, 0, 0],   # M3: C12
        [1, 0, 0, -1],  # M4: C11 - C22
        [0, 1, 0, 0],   # M5: C12
        [1, 0, 1, 0],   # M6: C11 + C21
        [1, 0, 0, -1]   # M7: C11 - C22 (wait, this is wrong)
    ]
    
    # Let me recheck Strassen:
    # C11 = M1 - M4 - M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6 - M7
    C_strassen = [
        [1, 0, 0, 0],   # M1: C11
        [0, 0, 1, 0],   # M2: C21
        [0, 1, 0, 0],   # M3: C12
        [-1, 0, 0, 1],  # M4: -C11 + C22
        [0, 1, 0, 0],   # M5: C12
        [-1, 0, 1, 0],  # M6: -C11 + C21
        [1, 0, 0, -1]   # M7: C11 - C22
    ]
    
    # Map block indices to actual (row, col) ranges
    # A block (a_block, col_offset): a_block 0=A11, 1=A12, 2=A21, 3=A22
    # A row: a_block // 2 (0 or 1), A col start: (a_block % 2) * 2
    def a_block_to_indices(block):
        row = block // 2  # 0 for A11/A12, 1 for A21/A22
        col_start = (block % 2) * 2  # 0 for A11/A21, 2 for A12/A22
        return row, col_start
    
    # B block (b_block): 0=B11, 1=B12, 2=B21, 3=B22
    # B row start: (b_block // 2) * 2, B col start: (b_block % 2) * 2
    def b_block_to_indices(block):
        row_start = (block // 2) * 2  # 0 for B11/B12, 2 for B21/B22
        col_start = (block % 2) * 2  # 0 for B11/B21, 2 for B12/B22
        return row_start, col_start
    
    # C block (c_block): 0=C11, 1=C12, 2=C21, 3=C22
    # C row: c_block // 2, C col start: (c_block % 2) * 2
    def c_block_to_indices(block):
        row = block // 2  # 0 for C11/C12, 1 for C21/C22
        col_start = (block % 2) * 2  # 0 for C11/C21, 2 for C12/C22
        return row, col_start
    
    # Now build the 28 rank-1 tensors for Part 1
    idx = 0
    for r in range(7):
        # For each Strassen term r, the A combination is a 1x2 vector
        # and B combination is a 2x2 matrix.
        # The inner multiplication (1x2) * (2x2) -> (1x2) has 4 rank-1 terms.
        for i in range(2):  # A index (row of the 1x2 A block)
            for j in range(2):  # B row index (row of the 2x2 B block)
                for k in range(2):  # B col index = C col index
                    # U contribution: A_block[i] for all blocks with nonzero coeff
                    u_vec = np.zeros(dim_u, dtype=np.int64)
                    for a_block in range(4):
                        coeff_a = A_strassen[r][a_block]
                        if coeff_a != 0:
                            row, col_start = a_block_to_indices(a_block)
                            # A_block[i] means: the i-th column of the 1x2 block
                            # So the actual A index is (row, col_start + i)
                            a_idx = row * m + (col_start + i)
                            u_vec[a_idx] += coeff_a
                    
                    # V contribution: B_block[j, k] for all blocks with nonzero coeff
                    v_vec = np.zeros(dim_v, dtype=np.int64)
                    for b_block in range(4):
                        coeff_b = B_strassen[r][b_block][j][k]
                        if coeff_b != 0:
                            row_start, col_start = b_block_to_indices(b_block)
                            # B_block[j, k] means: element (j, k) of the 2x2 block
                            # So the actual B index is (row_start + j, col_start + k)
                            b_idx = (row_start + j) * p + (col_start + k)
                            v_vec[b_idx] += coeff_b
                    
                    # W contribution: C_block[i, k] for all blocks with nonzero coeff
                    # Note: C_block is 1x2, so the element is at column k
                    # Wait, the inner multiplication: u (1x2) * v (2x2) -> w (1x2)
                    # w[k] = sum_i u[i] * v[i][k]
                    # So the C output column is k, and the C row is determined by the A row
                    # But the A row here is i (0 or 1), which maps to C row 0 or 1
                    # Actually, the A block row determines the C block row
                    # A block row 0 -> C block row 0, A block row 1 -> C block row 1
                    w_vec = np.zeros(dim_w, dtype=np.int64)
                    for c_block in range(4):
                        coeff_c = C_strassen[r][c_block]
                        if coeff_c != 0:
                            row_c, col_start_c = c_block_to_indices(c_block)
                            # C_block is 1x2, element at column k
                            # The actual C index: row_c, col_start_c + k
                            c_idx = row_c * n + (col_start_c + k)
                            # But we need to check: does the A row i match the C row?
                            # A block row i: 0->A11/A12 (row 0), 1->A21/A22 (row 1)
                            # C block row: 0->C11/C12 (row 0), 1->C21/C22 (row 1)
                            # The A block row determines which C block row
                            a_row = i  # 0 or 1
                            c_block_row = c_block // 2
                            if a_row == c_block_row:
                                w_vec[c_idx] += coeff_c
                    
                    # Check if this rank-1 term is non-zero
                    if np.any(u_vec) and np.any(v_vec) and np.any(w_vec):
                        U[:, idx] = u_vec
                        V[:, idx] = v_vec
                        W[:, idx] = w_vec
                        idx += 1
    
    # Part 2: 2x4 * 4x1 = 2x1, schoolbook rank 8
    for i in range(n):
        for j in range(m):
            a_idx = i * m + j
            b_idx = j * p + 4  # k=4 (last column)
            c_idx = 4 * n + i  # k=4, so c = 4*2 + i = 8 + i
            U[a_idx, idx] = 1
            V[b_idx, idx] = 1
            W[c_idx, idx] = 1
            idx += 1
    
    # idx should be 36
    assert idx == rank, f"Expected rank {rank}, got {idx}"
    
    # Verify the decomposition
    T_target = np.zeros((dim_u, dim_v, dim_w), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1
    
    T_decomp = np.zeros((dim_u, dim_v, dim_w), dtype=np.int64)
    for r in range(rank):
        T_decomp += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    
    if not np.array_equal(T_decomp, T_target):
        # Fallback to schoolbook rank 40
        U = np.zeros((dim_u, 40), dtype=np.int64)
        V = np.zeros((dim_v, 40), dtype=np.int64)
        W = np.zeros((dim_w, 40), dtype=np.int64)
        idx2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, idx2] = 1
                    V[b, idx2] = 1
                    W[c, idx2] = 1
                    idx2 += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
