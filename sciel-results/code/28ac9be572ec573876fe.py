import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W, n=3, m=4, p=8, tol=1e-9):
    """Verify that U, V, W form a valid decomposition."""
    rank = len(U[0]) if U else 0
    # Check shapes
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    if any(len(row) != rank for row in U):
        return False
    if any(len(row) != rank for row in V):
        return False
    if any(len(row) != rank for row in W):
        return False
    
    # Test with random matrices
    rng = np.random.default_rng(12345)
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    U_3d = U_np.reshape(n, m, rank)
    V_3d = V_np.reshape(m, p, rank)
    W_3d = W_np.reshape(p, n, rank)
    
    for _ in range(3):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C = A @ B
        
        # C_dec[i, k] = sum_c W[k, i, c] * ( sum_j U[i, j, c] * A[i, j] * V[j, k, c] )
        M = U_3d * A[:, :, np.newaxis]
        N = V_3d * B[:, :, np.newaxis]
        S = np.einsum('ijc,jkc->ikc', M, N)
        S_T = np.transpose(S, (1, 0, 2)) # shape (p, n, rank)
        C_dec = np.sum(S_T * W_3d, axis=2)
        
        if np.max(np.abs(C - C_dec)) > tol:
            return False
    return True

def _to_list(mat):
    if isinstance(mat, np.ndarray):
        mat = mat.tolist()
    return [[int(round(v)) if abs(v - round(v)) < 1e-9 else (round(v * 2) / 2 if abs(v * 2 - round(v * 2)) < 1e-9 else v) for v in row] for row in mat]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # --- Structural Block Decomposition Strategy ---
    # We partition the 3x4 matrix A and 4x8 matrix B.
    # A is 3x4. We can view the 3 rows as 1+2. The 4 cols as 2+2.
    # B is 4x8. We can view the 4 rows as 2+2. The 8 cols as 4+4.
    
    # Let's try a specific partition:
    # A = [ A11 (1x2)  A12 (1x2) ]
    #     [ A21 (2x2)  A22 (2x2) ]
    #     [ A31 (2x2)  A32 (2x2) ] -- Wait, 3 rows.
    # Let's partition rows of A into 1 and 2.
    # A_top (1x4), A_bottom (2x4).
    # Partition cols of A into 2 and 2.
    # A_top_left (1x2), A_top_right (1x2)
    # A_bot_left (2x2), A_bot_right (2x2)
    
    # B is 4x8. Rows 4, Cols 8.
    # Partition rows of B into 2 and 2 (matching cols of A).
    # B_top (2x8), B_bot (2x8).
    # Partition cols of B into 4 and 4.
    # B_top_left (2x4), B_top_right (2x4)
    # B_bot_left (2x4), B_bot_right (2x4)
    
    # Product C = A @ B (3x8)
    # C = [ A_top_left @ B_top_left + A_top_right @ B_bot_left ]  (1x8)
    #     [ A_bot_left @ B_top_left + A_bot_right @ B_bot_left ]  (2x8)
    #     [ A_bot_left @ B_top_left + A_bot_right @ B_bot_left ]  (2x8) -- Wait, row indices.
    
    # Let's be precise with indices.
    # A indices: i in {0, 1, 2}, j in {0, 1, 2, 3}
    # B indices: j in {0, 1, 2, 3}, k in {0, ..., 7}
    
    # Partition A rows: I1 = {0}, I2 = {1, 2}
    # Partition A cols: J1 = {0, 1}, J2 = {2, 3}
    # Partition B rows: J1 = {0, 1}, J2 = {2, 3} (must match A cols)
    # Partition B cols: K1 = {0, 1, 2, 3}, K2 = {4, 5, 6, 7}
    
    # Blocks of A:
    # A_11: rows I1, cols J1 -> 1x2
    # A_12: rows I1, cols J2 -> 1x2
    # A_21: rows I2, cols J1 -> 2x2
    # A_22: rows I2, cols J2 -> 2x2
    
    # Blocks of B:
    # B_11: rows J1, cols K1 -> 2x4
    # B_12: rows J1, cols K2 -> 2x4
    # B_21: rows J2, cols K1 -> 2x4
    # B_22: rows J2, cols K2 -> 2x4
    
    # Blocks of C (3x8):
    # C_11: rows I1, cols K1 -> 1x4. Formula: A_11 @ B_11 + A_12 @ B_21
    # C_12: rows I1, cols K2 -> 1x4. Formula: A_11 @ B_12 + A_12 @ B_22
    # C_21: rows I2, cols K1 -> 2x4. Formula: A_21 @ B_11 + A_22 @ B_21
    # C_22: rows I2, cols K2 -> 2x4. Formula: A_21 @ B_12 + A_22 @ B_22
    
    # We need to decompose these multiplications.
    # A_11 (1x2) @ B_11 (2x4): Rank 2 (schoolbook, since 1x2x4 min rank is 2? No, 1x2x4 tensor rank is 2? 
    # T_{ijk} = delta_{j,k}? No.
    # 1x2x4: A is 1x2, B is 2x4. C is 1x4.
    # C_k = A_0 B_{0k} + A_1 B_{1k}.
    # This is sum of 2 terms. Rank 2.
    
    # A_12 (1x2) @ B_21 (2x4): Rank 2.
    # A_21 (2x2) @ B_11 (2x4): 
    #   B_11 is 2x4. We can split B_11 cols into 2x2 and 2x2.
    #   A_21 @ B_11_left (2x2) + A_21 @ B_11_right (2x2).
    #   2x2 @ 2x2 has rank 7 (Strassen).
    #   So A_21 @ B_11 can be done in 7+7=14? Or is there a better way?
    #   Actually, 2x2 @ 2x4 tensor. 
    #   Let's just use Strassen for the 2x2 parts.
    
    # Let's build the decomposition term by term.
    # We will create lists of (U_row_idx, V_row_idx, W_row_idx, coeff) for each term.
    # Then sum them up.
    
    terms = []
    
    # Helper to add a block multiplication term
    # A_block is (h x w), B_block is (w x z)
    # We need to map global indices.
    
    # We will manually construct the 14 terms for 2x2 @ 2x2 using Strassen,
    # and 2 terms for 1x2 @ 2x2 (or 1x2 @ 2x4 split).
    
    # Let's define the sub-blocks clearly.
    # A_11 (1x2): global rows [0], cols [0,1]
    # A_12 (1x2): global rows [0], cols [2,3]
    # A_21 (2x2): global rows [1,2], cols [0,1]
    # A_22 (2x2): global rows [1,2], cols [2,3]
    
    # B_11 (2x4): global rows [0,1], cols [0,1,2,3]
    # B_12 (2x4): global rows [0,1], cols [4,5,6,7]
    # B_21 (2x4): global rows [2,3], cols [0,1,2,3]
    # B_22 (2x4): global rows [2,3], cols [4,5,6,7]
    
    # We need to decompose:
    # 1. A_11 @ B_11 (1x2 @ 2x4) -> C_11_left (1x4)
    # 2. A_12 @ B_21 (1x2 @ 2x4) -> C_11_right (1x4)
    # 3. A_11 @ B_12 (1x2 @ 2x4) -> C_12_left (1x4)
    # 4. A_12 @ B_22 (1x2 @ 2x4) -> C_12_right (1x4)
    # 5. A_21 @ B_11 (2x2 @ 2x4) -> C_21_left (2x4)
    # 6. A_22 @ B_21 (2x2 @ 2x4) -> C_21_right (2x4)
    # 7. A_21 @ B_12 (2x2 @ 2x4) -> C_22_left (2x4)
    # 8. A_22 @ B_22 (2x2 @ 2x4) -> C_22_right (2x4)
    
    # Rank estimation:
    # 1x2 @ 2x4: Rank 2. (4 blocks * 2 = 8)
    # 2x2 @ 2x4: Split B into two 2x2.
    #   2x2 @ 2x2 is Rank 7.
    #   So 2x2 @ 2x4 is Rank 14. (4 blocks * 14 = 56)
    # Total Rank = 8 + 56 = 64.
    
    # This is significantly better than 96 (schoolbook).
    
    # Let's implement the 1x2 @ 2x4 decomposition.
    # A is 1x2: [a0, a1]
    # B is 2x4: [b00, b01, b02, b03; b10, b11, b12, b13]
    # C is 1x4: [a0*b00 + a1*b10, ...]
    # Term 1: a0 * B_row0. U: a0, V: B_row0, W: 1.
    # Term 2: a1 * B_row1. U: a1, V: B_row1, W: 1.
    
    # Let's implement the 2x2 @ 2x2 Strassen decomposition.
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b10)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b10
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # c00 = M1 - M4 + M6 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M3 + M2 + M5
    
    # We will generate the (U, V, W) contributions for each sub-block.
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        terms.append((u_idx, v_idx, w_idx, coeff))
    
    # 1. A_11 @ B_11 (1x2 @ 2x4)
    # A_11 rows: [0], cols: [0,1] -> global U indices: 0, 1
    # B_11 rows: [0,1], cols: [0,1,2,3] -> global V indices: 0,1,2,3 (row 0) and 4,5,6,7 (row 1)
    # C_11_left: rows [0], cols [0,1,2,3] -> global W indices: 0,1,2,3
    
    # Term 1: a0 * B_row0
    for k in range(4):
        add_term(0, k, k, 1) # U[0] * V[k] * W[k]
    # Term 2: a1 * B_row1
    for k in range(4):
        add_term(1, 4+k, k, 1) # U[1] * V[4+k] * W[k]
        
    # 2. A_12 @ B_21 (1x2 @ 2x4)
    # A_12 rows: [0], cols: [2,3] -> global U indices: 2, 3
    # B_21 rows: [2,3], cols: [0,1,2,3] -> global V indices: 8,9,10,11 (row 2) and 12,13,14,15 (row 3)
    # C_11_right: rows [0], cols [4,5,6,7] -> global W indices: 4,5,6,7
    
    # Term 1: a2 * B_row2
    for k in range(4):
        add_term(2, 8+k, 4+k, 1)
    # Term 2: a3 * B_row3
    for k in range(4):
        add_term(3, 12+k, 4+k, 1)
        
    # 3. A_11 @ B_12 (1x2 @ 2x4)
    # A_11 rows: [0], cols: [0,1] -> U: 0, 1
    # B_12 rows: [0,1], cols: [4,5,6,7] -> V: 16,17,18,19 (row 0) and 20,21,22,23 (row 1)
    # C_12_left: rows [0], cols [8,9,10,11] -> W: 8,9,10,11
    
    for k in range(4):
        add_term(0, 16+k, 8+k, 1)
    for k in range(4):
        add_term(1, 20+k, 8+k, 1)
        
    # 4. A_12 @ B_22 (1x2 @ 2x4)
    # A_12 rows: [0], cols: [2,3] -> U: 2, 3
    # B_22 rows: [2,3], cols: [4,5,6,7] -> V: 24,25,26,27 (row 2) and 28,29,30,31 (row 3)
    # C_12_right: rows [0], cols [12,13,14,15] -> W: 12,13,14,15
    
    for k in range(4):
        add_term(2, 24+k, 12+k, 1)
    for k in range(4):
        add_term(3, 28+k, 12+k, 1)
        
    # Now the 2x2 @ 2x4 blocks.
    # We need a helper to add Strassen terms for a 2x2 @ 2x2 block.
    
    def add_strassen_2x2(u_base, v_base, w_base, row_offsets, col_offsets):
        # u_base: global U index of top-left element of A_block (2x2)
        # v_base: global V index of top-left element of B_block (2x2)
        # w_base: global W index of top-left element of C_block (2x2)
        # row_offsets: [r0, r1] global row indices for A/C
        # col_offsets: [c0, c1] global col indices for A (cols of A = rows of B)
        #                [k0, k1] global col indices for B/C
        
        # A indices:
        # a00: u_base
        # a01: u_base + 1
        # a10: u_base + 2
        # a11: u_base + 3
        
        # B indices:
        # b00: v_base
        # b01: v_base + 1
        # b10: v_base + 2
        # b11: v_base + 3
        
        # C indices:
        # c00: w_base
        # c01: w_base + 1
        # c10: w_base + 2
        # c11: w_base + 3
        
        # M1 = (a00+a11)(b00+b11)
        # We need to represent sums.
        # U term for a00+a11: We can't just add U indices in a single term.
        # We must expand: a00*b00 + a00*b11 + a11*b00 + a11*b11
        # This would be 4 terms. But Strassen uses 7 terms total for the product.
        # The "terms" in Strassen are linear combinations of inputs multiplied together.
        # In the tensor decomposition, each term is U_i * V_j * W_k.
        # If U_i is a sum, e.g., U_i = a00 + a11, that's not allowed in a single basis vector.
        # Wait, the basis vectors U, V, W can have coefficients!
        # U is a matrix of size (n*m) x Rank.
        # Each column c of U is a vector in R^{n*m}.
        # So U[i, c] can be 1, -1, 0, etc.
        # So we CAN have a term where U[:, c] has 1 at a00 and 1 at a11.
        # Yes! That's how Strassen works in this representation.
        
        # Let's define the 7 Strassen terms for this block.
        # We will append to the global 'terms' list.
        
        # M1: U has 1 at a00, 1 at a11. V has 1 at b00, 1 at b11.
        #      W has 1 at c00, 1 at c11.
        #      Wait, W contribution is to c00 and c11?
        #      c00 = M1 - M4 + M6 + M7
        #      c11 = M1 - M3 + M2 + M5
        #      So M1 contributes +1 to c00 and +1 to c11.
        
        # Let's map the Strassen variables to (U_vec, V_vec, W_vec)
        
        # M1:
        u_vec = [0]* (n*m)
        u_vec[row_offsets[0]*m + col_offsets[0]] = 1 # a00
        u_vec[row_offsets[1]*m + col_offsets[1]] = 1 # a11
        v_vec = [0]* (m*p)
        v_vec[col_offsets[0]*p + row_offsets[0]] = 1 # b00 (B row 0, col 0)
        # Wait, B indices.
        # B is 4x8.
        # B_block is 2x2.
        # B rows are global rows r0, r1.
        # B cols are global cols k0, k1.
        # b00 is at B[r0, k0]. Index: r0*p + k0.
        # b11 is at B[r1, k1]. Index: r1*p + k1.
        
        # Let's re-evaluate indices for add_strassen_2x2.
        # Inputs:
        # A_block rows: ar0, ar1. Cols: ac0, ac1.
        # B_block rows: br0, br1. Cols: bc0, bc1.
        # C_block rows: cr0, cr1. Cols: cc0, cc1.
        
        # Note: ac0=br0, ac1=br1 (inner dimension match).
        
        # M1: (a00+a11)(b00+b11)
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[ar0*m + ac0] = 1; u[ar1*m + ac1] = 1
        v[br0*p + bc0] = 1; v[br1*p + bc1] = 1
        w[cc0*n + cr0] = 1; w[cc1*n + cr1] = 1
        terms.append((u, v, w))
        
        # M2: (a10+a11)b00
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[ar1*m + ac0] = 1; u[ar1*m + ac1] = 1
        v[br0*p + bc0] = 1
        w[cc0*n + cr1] = 1; w[cc1*n + cr1] = 1
        terms.append((u, v, w))
        
        # M3: a00(b01-b10)
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[ar0*m + ac0] = 1
        v[br0*p + bc1] = 1; v[br1*p + bc0] = -1
        w[cc1*n + cr0] = 1; w[cc1*n + cr1] = -1
        terms.append((u, v, w))
        
        # M4: a11(b10-b00)
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[ar1*m + ac1] = 1
        v[br1*p + bc0] = 1; v[br0*p + bc0] = -1
        w[cc0*n + cr0] = -1; w[cc0*n + cr1] = 1
        terms.append((u, v, w))
        
        # M5: (a00+a01)b10
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[ar0*m + ac0] = 1; u[ar0*m + ac1] = 1
        v[br1*p + bc0] = 1
        w[cc1*n + cr0] = 1; w[cc1*n + cr1] = 1
        terms.append((u, v, w))
        
        # M6: (a10-a00)(b00+b01)
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[ar1*m + ac0] = 1; u[ar0*m + ac0] = -1
        v[br0*p + bc0] = 1; v[br0*p + bc1] = 1
        w[cc0*n + cr0] = 1
        terms.append((u, v, w))
        
        # M7: (a01-a11)(b10+b11)
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[ar0*m + ac1] = 1; u[ar1*m + ac1] = -1
        v[br1*p + bc0] = 1; v[br1*p + bc1] = 1
        w[cc0*n + cr0] = 1; w[cc1*n + cr1] = 1
        terms.append((u, v, w))
        
    # 5. A_21 @ B_11 (2x2 @ 2x4)
    # A_21: rows [1,2], cols [0,1]
    # B_11: rows [0,1], cols [0,1,2,3]
    # C_21_left: rows [1,2], cols [0,1,2,3]
    
    # Split B_11 into B_11_L (cols 0,1) and B_11_R (cols 2,3)
    # B_11_L: rows [0,1], cols [0,1]
    # B_11_R: rows [0,1], cols [2,3]
    
    # A_21 @ B_11_L -> C_21_LL (rows [1,2], cols [0,1])
    add_strassen_2x2(
        ar0=1, ar1=2, ac0=0, ac1=1,
        br0=0, br1=1, bc0=0, bc1=1,
        cr0=1, cr1=2, cc0=0, cc1=1
    )
    
    # A_21 @ B_11_R -> C_21_LR (rows [1,2], cols [2,3])
    add_strassen_2x2(
        ar0=1, ar1=2, ac0=0, ac1=1,
        br0=0, br1=1, bc0=2, bc1=3,
        cr0=1, cr1=2, cc0=2, cc1=3
    )
    
    # 6. A_22 @ B_21 (2x2 @ 2x4)
    # A_22: rows [1,2], cols [2,3]
    # B_21: rows [2,3], cols [0,1,2,3]
    # C_21_right: rows [1,2], cols [4,5,6,7]
    
    # Split B_21 into B_21_L (cols 0,1) and B_21_R (cols 2,3)
    
    # A_22 @ B_21_L -> C_21_RL (rows [1,2], cols [4,5])
    add_strassen_2x2(
        ar0=1, ar1=2, ac0=2, ac1=3,
        br0=2, br1=3, bc0=0, bc1=1,
        cr0=1, cr1=2, cc0=4, cc1=5
    )
    
    # A_22 @ B_21_R -> C_21_RR (rows [1,2], cols [6,7])
    add_strassen_2x2(
        ar0=1, ar1=2, ac0=2, ac1=3,
        br0=2, br1=3, bc0=2, bc1=3,
        cr0=1, cr1=2, cc0=6, cc1=7
    )
    
    # 7. A_21 @ B_12 (2x2 @ 2x4)
    # A_21: rows [1,2], cols [0,1]
    # B_12: rows [0,1], cols [4,5,6,7]
    # C_22_left: rows [1,2], cols [8,9,10,11]
    
    # Split B_12 into B_12_L (cols 4,5) and B_12_R (cols 6,7)
    
    # A_21 @ B_12_L -> C_22_LL (rows [1,2], cols [8,9])
    add_strassen_2x2(
        ar0=1, ar1=2, ac0=0, ac1=1,
        br0=0, br1=1, bc0=4, bc1=5,
        cr0=1, cr1=2, cc0=8, cc1=9
    )
    
    # A_21 @ B_12_R -> C_22_LR (rows [1,2], cols [10,11])
    add_strassen_2x2(
        ar0=1, ar1=2, ac0=0, ac1=1,
        br0=0, br1=1, bc0=6, bc1=7,
        cr0=1, cr1=2, cc0=10, cc1=11
    )
    
    # 8. A_22 @ B_22 (2x2 @ 2x4)
    # A_22: rows [1,2], cols [2,3]
    # B_22: rows [2,3], cols [4,5,6,7]
    # C_22_right: rows [1,2], cols [12,13,14,15]
    
    # Split B_22 into B_22_L (cols 4,5) and B_22_R (cols 6,7)
    
    # A_22 @ B_22_L -> C_22_RL (rows [1,2], cols [12,13])
    add_strassen_2x2(
        ar0=1, ar1=2, ac0=2, ac1=3,
        br0=2, br1=3, bc0=4, bc1=5,
        cr0=1, cr1=2, cc0=12, cc1=13
    )
    
    # A_22 @ B_22_R -> C_22_RR (rows [1,2], cols [14,15])
    add_strassen_2x2(
        ar0=1, ar1=2, ac0=2, ac1=3,
        br0=2, br1=3, bc0=6, bc1=7,
        cr0=1, cr1=2, cc0=14, cc1=15
    )
    
    # Now we have a list of terms.
    # Each term is (U_vec, V_vec, W_vec).
    # We need to combine them into matrices U, V, W of shape (n*m, R), (m*p, R), (p*n, R).
    # R = len(terms).
    
    R = len(terms)
    U_mat = np.zeros((n*m, R))
    V_mat = np.zeros((m*p, R))
    W_mat = np.zeros((p*n, R))
    
    for c, (u, v, w) in enumerate(terms):
        U_mat[:, c] = u
        V_mat[:, c] = v
        W_mat[:, c] = w
        
    # Convert to lists
    U_list = _to_list(U_mat)
    V_list = _to_list(V_mat)
    W_list = _to_list(W_mat)
    
    # Verify
    if _verify(U_list, V_list, W_list, n, m, p):
        best = {"U": U_list, "V": V_list, "W": W_list}
        best_rank = R
        
    # Return best
    return best

