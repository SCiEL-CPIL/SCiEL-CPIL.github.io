"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    # Block-decomposed construction:
    # 2x3 * 3x8 = 2x8
    # Split B into B_1 (3x3), B_2 (3x3), B_3 (3x2)
    # C = A*B_1 + A*B_2 + A*B_3
    # A is 2x3. For each B block:
    #   A (2x3) * B_block (3xp_block) -> 2xp_block
    # Split A into A_left (2x2) and A_right (2x1)
    # For B_block 3xp: split into B_left (3x(p-1)) and B_right (3x1)
    #   A_left*B_left: 2x2 * (p-1)x(p-1) -> use Strassen for 2x2*2x2 part + remaining
    #   Actually let's think more carefully.
    #
    # For 2x3 * 3x8:
    # The tensor T[i,j,k] where i in [0,2), j in [0,3), k in [0,8)
    # T[i,j,k] = 1 if the (i,j) element of A multiplies the (j,k) element of B
    # i.e., C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Schoolbook: 2*3*8 = 48 terms.
    #
    # Strategy: split the 8 columns of B into three blocks: 3, 3, 2 columns.
    # Block 1: B[:, 0:3] is 3x3
    # Block 2: B[:, 3:6] is 3x3
    # Block 3: B[:, 6:8] is 3x2
    #
    # For 2x3 * 3x3 (each of blocks 1 and 2):
    #   Split A = [A12 (2x2), A2 (2x1)]
    #   Split B = [B12 (3x2); B3 (1x3)]  -- no wait, B is 3x3
    #   B rows: B[0:2,:] is 2x3, B[2,:] is 1x3
    #   C = A12 * B[0:2,:] + A2 * B[2,:]
    #   A12 * B[0:2,:]: 2x2 * 2x3
    #   A2 * B[2,:]: 2x1 * 1x3
    #
    # For 2x2 * 2x3:
    #   Split the 3 columns into 2 and 1
    #   2x2 * 2x2: rank 7 (Strassen)
    #   2x2 * 2x1: rank 4 (schoolbook, since one side is 2x1)
    #   2x1 * 1x3: rank 1? No, 2x1 * 1x3 = 2x3. 
    #   Actually A2 is 2x1, B[2,:] is 1x3. Product is 2x3.
    #   C[i,k] = A2[i] * B[2,k]
    #   This is sum over... no, it's a single product per (i,k).
    #   In tensor form: u_i * v_j * w_k where j is fixed at 2.
    #   u = (1,1), v = (0,0,1), w = (1,1,1) -> rank 1 term.
    #   Wait: u_i * v_j * w_k. v_j = delta_{j,2}. u_i = 1 for i in {0,1}. w_k = 1 for k in {0,1,2}.
    #   Yes, rank 1.
    #
    # So 2x2 * 2x3: 
    #   Split into 2x2*2x2 (rank 7) + 2x2*2x1 (rank 4) = rank 11
    #   Then add 2x1*1x3 (rank 1) = total rank 12 per 2x3*3x3 block.
    #
    # For 2x3 * 3x2 (block 3):
    #   A = [A12 (2x2), A2 (2x1)]
    #   B is 3x2: B[0:2,:] is 2x2, B[2,:] is 1x2
    #   C = A12 * B[0:2,:] + A2 * B[2,:]
    #   A12 * B[0:2,:]: 2x2 * 2x2, rank 7
    #   A2 * B[2,:]: 2x1 * 1x2, rank 1
    #   Total: rank 8 per 2x3*3x2 block.
    #
    # Total: 12 + 12 + 8 = 32.
    #
    # Let's verify the Strassen 2x2*2x2 decomposition (rank 7):
    # Standard Strassen:
    #   M1 = (A11+A22)(B11+B22)
    #   M2 = (A21+A22)B11
    #   M3 = A11(B21-B22)
    #   M4 = A22(B11-B21)
    #   M5 = (A11+A12)B22
    #   M6 = (A21-A11)(B11+B12)
    #   M7 = (A12-A22)(B21+B22)
    #   C11 = M1 - M4 + M5 - M7
    #   C12 = M3 + M5
    #   C21 = M2 + M4
    #   C22 = M1 - M2 + M3 + M6
    #
    # In tensor form, each M_r = (u_r . A_sub) * (v_r . B_sub) contributes to C entries.
    # For 2x2*2x2, the indices are i,j in {0,1} for A (sub-indices), and k,l in {0,1} for B sub-indices.
    # C_sub[i,l] = sum_{j,k} A_sub[i,j] * B_sub[j,k] * T[i,j,k,l]
    #
    # Let's define the 7 terms for 2x2*2x2. Each term r has vectors a_r (2), b_r (2), and output coefficients c_r (2x2 -> 4 values).
    #
    # M1: a=(1,1) [A11+A22 means i=0,j=0 and i=1,j=1... wait]
    # Actually, Strassen operates on the 2x2 submatrices. Let me redefine.
    # A_sub = [[a00, a01], [a10, a11]]
    # B_sub = [[b00, b01], [b10, b11]]
    #
    # M1 = (a00+a11)(b00+b11)
    #   a_vec for this: coefficient of A terms: a00*1 + a11*1, others 0. So a_r = (1, 0, 0, 1) in flattened (row-major: a00,a01,a10,a11)
    #   Wait, we need to be careful. The U matrix for A has entries U[i,j, r]. For the sub-problem, i,j are local indices.
    #   Let me use local indices for the 2x2 subproblem.
    #
    # For M1: A part is a00 + a11. So U_local[0,0,r]=1, U_local[1,1,r]=1, others 0.
    #          B part is b00 + b11. V_local[0,0,r]=1, V_local[1,1,r]=1, others 0.
    #          C part: C11 += 1, C22 += 1. So W_local[0,0,r]=1, W_local[1,1,r]=1, others 0.
    #
    # M2 = (a10+a11)b00
    #   A: a10+a11. U_local[1,0,r]=1, U_local[1,1,r]=1.
    #   B: b00. V_local[0,0,r]=1.
    #   C: C21 += 1. W_local[1,0,r]=1.
    #
    # M3 = a00(b10-b11)
    #   A: a00. U_local[0,0,r]=1.
    #   B: b10-b11. V_local[1,0,r]=1, V_local[1,1,r]=-1.
    #   C: C12 += 1. W_local[0,1,r]=1.
    #
    # M4 = a11(b00-b10)
    #   A: a11. U_local[1,1,r]=1.
    #   B: b00-b10. V_local[0,0,r]=1, V_local[1,0,r]=-1.
    #   C: C11 -= 1. W_local[0,0,r]=-1.
    #
    # M5 = (a00+a01)b11
    #   A: a00+a01. U_local[0,0,r]=1, U_local[0,1,r]=1.
    #   B: b11. V_local[1,1,r]=1.
    #   C: C11 += 1, C12 += 1. W_local[0,0,r]=1, W_local[0,1,r]=1.
    #
    # M6 = (a10-a00)(b00+b01)
    #   A: a10-a00. U_local[1,0,r]=1, U_local[0,0,r]=-1.
    #   B: b00+b01. V_local[0,0,r]=1, V_local[0,1,r]=1.
    #   C: C22 += 1. W_local[1,1,r]=1.
    #
    # M7 = (a01-a11)(b10+b11)
    #   A: a01-a11. U_local[0,1,r]=1, U_local[1,1,r]=-1.
    #   B: b10+b11. V_local[1,0,r]=1, V_local[1,1,r]=1.
    #   C: C11 -= 1. W_local[0,0,r]=-1.
    #
    # Now for 2x2 * 2x1 (rank 4):
    #   A_sub is 2x2, B_sub is 2x1.
    #   C[i,0] = a_i0*b_0 + a_i1*b_1
    #   This is just schoolbook: 4 terms.
    #   Term r: U_local[i,j,r]=delta, V_local[j,0,r]=delta, W_local[i,0,r]=1.
    #   4 terms: (i,j) in {(0,0),(0,1),(1,0),(1,1)}.
    #
    # For 2x2 * 2x2 with the W convention: W[k*n+i, r] where n=2 (output rows), k is the B column index.
    # Wait, the global W is indexed as W[k*n+i, r] where i is the output row (0..n-1) and k is the B column index.
    # In the subproblem, the local W index is (local_k, local_i) -> global offset.
    #
    # Let me now build the full 2x3*3x8 decomposition.
    # Global indices: i in 0..1 (A rows), j in 0..2 (A cols / B rows), k in 0..7 (B cols).
    # U has shape (n*m, rank) = (2*3, rank) = (6, rank). U[i*m+j, r]
    # V has shape (m*p, rank) = (3*8, rank) = (24, rank). V[j*p+k, r]
    # W has shape (p*n, rank) = (8*2, rank) = (16, rank). W[k*n+i, r]
    #
    # Block 1: columns 0,1,2 of B (k_local = 0,1,2 -> k_global = 0,1,2)
    #   A = [A12 (2x2, cols 0,1), A2 (2x1, col 2)]
    #   B1 = [B12 (2x3, rows 0,1), B3 (1x3, row 2)]
    #   C = A12*B12 + A2*B3
    #   A12*B12: 2x2 * 2x3. Split B12 into B12a (2x2, cols 0,1) and B12b (2x1, col 2).
    #   A12*B12a: 2x2*2x2, rank 7.
    #   A12*B12b: 2x2*2x1, rank 4.
    #   A2*B3: 2x1*1x3, rank 1.
    #   Total for block 1: 7+4+1 = 12.
    #
    # Block 2: columns 3,4,5 of B (k_global = 3,4,5)
    #   Same structure, rank 12.
    #
    # Block 3: columns 6,7 of B (k_global = 6,7)
    #   B3 = [B3a (2x2, rows 0,1), B3b (1x2, row 2)]
    #   C = A12*B3a + A2*B3b
    #   A12*B3a: 2x2*2x2, rank 7.
    #   A2*B3b: 2x1*1x2, rank 1.
    #   Total for block 3: 7+1 = 8.
    #
    # Grand total: 12+12+8 = 32.
    #
    # Now let's construct the U, V, W matrices.
    # We'll define a list of terms. Each term is (U_entry, V_entry, W_entry) where each is a dict or list of (index, value).
    
    import time
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    U_rows = n * m  # 6
    V_rows = m * p  # 24
    W_rows = p * n  # 16
    
    # We'll build lists of (row, col, value) for each matrix.
    U_entries = []
    V_entries = []
    W_entries = []
    
    # Helper to add a term
    term_id = 0
    def add_term(U_vals, V_vals, W_vals):
        """U_vals: dict {u_row: val}, V_vals: dict {v_row: val}, W_vals: dict {w_row: val}"""
        nonlocal term_id
        for u_row, val in U_vals.items():
            if val != 0:
                U_entries.append((u_row, term_id, val))
        for v_row, val in V_vals.items():
            if val != 0:
                V_entries.append((v_row, term_id, val))
        for w_row, val in W_vals.items():
            if val != 0:
                W_entries.append((w_row, term_id, val))
        term_id += 1
    
    # Strassen 2x2*2x2: 7 terms
    # Local indices: A[i,j] i,j in 0,1; B[j,k] j,k in 0,1; C[i,k] i,k in 0,1
    # We'll define the 7 terms with their contributions.
    def strassen_2x2_2x2(U_base, V_base, W_base, a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        U_base: function(u_local_i, u_local_j) -> global u_row
        V_base: function(v_local_j, v_local_k) -> global v_row
        W_base: function(w_local_k, w_local_i) -> global w_row
        """
        # M1: (a00+a11)(b00+b11)
        add_term(
            {U_base(0,0): 1, U_base(1,1): 1},
            {V_base(0,0): 1, V_base(1,1): 1},
            {W_base(0,0): 1, W_base(1,1): 1}
        )
        # M2: (a10+a11)b00
        add_term(
            {U_base(1,0): 1, U_base(1,1): 1},
            {V_base(0,0): 1},
            {W_base(0,1): 1}  # C21 -> i=1,k=0 -> W[0*2+1, r] = W[1, r]
        )
        # M3: a00(b10-b11)
        add_term(
            {U_base(0,0): 1},
            {V_base(1,0): 1, V_base(1,1): -1},
            {W_base(1,0): 1}  # C12 -> i=0,k=1 -> W[1*2+0, r] = W[2, r]
        )
        # M4: a11(b00-b10)
        add_term(
            {U_base(1,1): 1},
            {V_base(0,0): 1, V_base(1,0): -1},
            {W_base(0,0): -1}  # C11 -> i=0,k=0 -> W[0, r]
        )
        # M5: (a00+a01)b11
        add_term(
            {U_base(0,0): 1, U_base(0,1): 1},
            {V_base(1,1): 1},
            {W_base(0,0): 1, W_base(1,0): 1}  # C11 and C12
        )
        # M6: (a10-a00)(b00+b01)
        add_term(
            {U_base(1,0): 1, U_base(0,0): -1},
            {V_base(0,0): 1, V_base(0,1): 1},
            {W_base(1,1): 1}  # C22 -> i=1,k=1 -> W[1*2+1, r] = W[3, r]
        )
        # M7: (a01-a11)(b10+b11)
        add_term(
            {U_base(0,1): 1, U_base(1,1): -1},
            {V_base(1,0): 1, V_base(1,1): 1},
            {W_base(0,0): -1}  # C11
        )
    
    # Schoolbook 2x2*2x1: 4 terms
    def schoolbook_2x2_2x1(U_base, V_base, W_base):
        """
        A is 2x2, B is 2x1. C is 2x1.
        U_base(i, j) -> u_row
        V_base(j, k) -> v_row, k=0 only
        W_base(k, i) -> w_row, k=0 only
        """
        for i in range(2):
            for j in range(2):
                add_term(
                    {U_base(i, j): 1},
                    {V_base(j, 0): 1},
                    {W_base(0, i): 1}
                )
    
    # Schoolbook 2x1*1x3: 1 term (rank 1)
    def rank1_2x1_1x3(U_base, V_base, W_base):
        """
        A is 2x1 (only col 0), B is 1x3 (only row 0).
        C[i,k] = a_i0 * b_0k
        U_base(i, j) -> u_row, j=0
        V_base(j, k) -> v_row, j=0
        W_base(k, i) -> w_row
        """
        add_term(
            {U_base(0, 0): 1, U_base(1, 0): 1},
            {V_base(0, 0): 1, V_base(0, 1): 1, V_base(0, 2): 1},
            {W_base(0, 0): 1, W_base(1, 0): 1, W_base(2, 0): 1,
             W_base(0, 1): 1, W_base(1, 1): 1, W_base(2, 1): 1}
        )
    
    # Rank 1 for 2x1*1x2
    def rank1_2x1_1x2(U_base, V_base, W_base):
        """
        A is 2x1, B is 1x2. C is 2x2.
        C[i,k] = a_i0 * b_0k
        """
        add_term(
            {U_base(0, 0): 1, U_base(1, 0): 1},
            {V_base(0, 0): 1, V_base(0, 1): 1},
            {W_base(0, 0): 1, W_base(1, 0): 1,
             W_base(0, 1): 1, W_base(1, 1): 1}
        )
    
    # Now build the full decomposition.
    # Block 1: B columns 0,1,2
    # A12: A[0:2, 0:2], A2: A[:, 2]
    # B12: B[0:2, 0:3], B3: B[2, 0:3]
    # A12*B12a: 2x2*2x2, B12a = B[0:2, 0:2]
    # A12*B12b: 2x2*2x1, B12b = B[0:2, 2:3]
    # A2*B3: 2x1*1x3
    
    # U_base for A12: U[i*3 + j, r] where i in 0,1 and j in 0,1
    def U_base_A12_block1(i, j):
        return i * m + j  # i*3+j
    def V_base_B12a_block1(j, k):
        return j * p + k  # j*8+k, j in 0,1, k in 0,1
    def W_base_C_block1(k, i):
        return k * n + i  # k*2+i, k in 0,1, i in 0,1
    
    strassen_2x2_2x2(U_base_A12_block1, V_base_B12a_block1, W_base_C_block1, 
                     0, 0, 0, 0, 0, 0)
    
    # A12*B12b: 2x2*2x1, B12b = B[0:2, 2] (col 2)
    def V_base_B12b_block1(j, k_local):
        # k_local is 0, maps to global k=2
        return j * p + (2 + k_local)
    def W_base_C_block1b(k_local, i):
        # k_local is 0, maps to global k=2
        return (2 + k_local) * n + i
    schoolbook_2x2_2x1(U_base_A12_block1, V_base_B12b_block1, W_base_C_block1b)
    
    # A2*B3: 2x1*1x3
    # A2 = A[:, 2], so U_base(i, 0) = i*3+2
    def U_base_A2_block1(i, j):
        return i * m + 2  # j is always 0
    def V_base_B3_block1(j, k):
        return 2 * p + k  # row 2 of B, k in 0,1,2
    def W_base_C_block1c(k, i):
        return k * n + i
    rank1_2x1_1x3(U_base_A2_block1, V_base_B3_block1, W_base_C_block1c)
    
    # Block 2: B columns 3,4,5
    def U_base_A12_block2(i, j):
        return i * m + j
    def V_base_B12a_block2(j, k):
        return j * p + (3 + k)  # k in 0,1 -> global 3,4
    def W_base_C_block2(k, i):
        return (3 + k) * n + i  # k in 0,1 -> global 3,4
    
    strassen_2x2_2x2(U_base_A12_block2, V_base_B12a_block2, W_base_C_block2,
                     0, 0, 0, 0, 0, 0)
    
    # A12*B12b: 2x2*2x1, B12b = B[0:2, 5] (col 5)
    def V_base_B12b_block2(j, k_local):
        return j * p + (5 + k_local)
    def W_base_C_block2b(k_local, i):
        return (5 + k_local) * n + i
    schoolbook_2x2_2x1(U_base_A12_block2, V_base_B12b_block2, W_base_C_block2b)
    
    # A2*B3: 2x1*1x3
    def U_base_A2_block2(i, j):
        return i * m + 2
    def V_base_B3_block2(j, k):
        return 2 * p + (3 + k)  # row 2, k in 0,1,2 -> global 3,4,5
    def W_base_C_block2c(k, i):
        return (3 + k) * n + i
    rank1_2x1_1x3(U_base_A2_block2, V_base_B3_block2, W_base_C_block2c)
    
    # Block 3: B columns 6,7
    # A12*B3a: 2x2*2x2, B3a = B[0:2, 6:8]
    def U_base_A12_block3(i, j):
        return i * m + j
    def V_base_B3a_block3(j, k):
        return j * p + (6 + k)  # k in 0,1 -> global 6,7
    def W_base_C_block3(k, i):
        return (6 + k) * n + i
    
    strassen_2x2_2x2(U_base_A12_block3, V_base_B3a_block3, W_base_C_block3,
                     0, 0, 0, 0, 0, 0)
    
    # A2*B3b: 2x1*1x2, B3b = B[2, 6:8]
    def U_base_A2_block3(i, j):
        return i * m + 2
    def V_base_B3b_block3(j, k):
        return 2 * p + (6 + k)  # k in 0,1 -> global 6,7
    def W_base_C_block3b(k, i):
        return (6 + k) * n + i
    rank1_2x1_1x2(U_base_A2_block3, V_base_B3b_block3, W_base_C_block3b)
    
    # Now convert entries to matrices.
    rank = term_id
    
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for u_row, r, val in U_entries:
        U[u_row][r] = val
    for v_row, r, val in V_entries:
        V[v_row][r] = val
    for w_row, r, val in W_entries:
        W[w_row][r] = val
    
    if time.time() > deadline:
        # Return whatever we have; should be fine since construction is fast
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
