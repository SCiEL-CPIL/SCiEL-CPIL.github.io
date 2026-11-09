import time
import sys

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=7
    
    # Initialize U, V, W
    # We will build them as lists of lists
    # We don't know the final rank yet, so we'll use a list of tuples (i, val) or similar
    # Or just build the matrices and track rank.
    
    # Let's use the block decomposition strategy.
    # A is 3x4. B is 4x7.
    
    # A blocks:
    # A11: rows 0,1; cols 0,1 (2x2)
    # A12: rows 0,1; cols 2,3 (2x2)
    # A21: row 2; cols 0,1 (1x2)
    # A22: row 2; cols 2,3 (1x2)
    
    # B blocks:
    # B11: rows 0,1; cols 0,1 (2x2)
    # B12: rows 0,1; cols 2,3 (2x2)
    # B13: rows 0,1; cols 4,5,6 (2x3)
    # B21: rows 2,3; cols 0,1 (2x2)
    # B22: rows 2,3; cols 2,3 (2x2)
    # B23: rows 2,3; cols 4,5,6 (2x3)
    
    # We will accumulate U, V, W
    U_rows = []
    V_rows = []
    W_rows = []
    
    # Helper to add a term
    # term: (u_index, v_index, w_index, coeff_u, coeff_v, coeff_w)
    # Actually, for each rank-1 component, we add 1 to U[u_idx, c], V[v_idx, c], W[w_idx, c]
    # and multiply by coefficients if needed.
    # Strassen terms have +/- 1 coefficients.
    
    current_rank = 0
    
    def add_term(u_indices, v_indices, w_indices, coeffs):
        """
        u_indices: list of global row indices for A
        v_indices: list of global row indices for B (which are cols of A)
        w_indices: list of global row indices for C (which are rows of B, cols of A output? No)
        
        Wait, the tensor is T[i, j, k].
        U[i, c], V[j, c], W[k, c].
        
        For a block multiplication A_block * B_block:
        A_block is at A[r_a + i, c_a + j]
        B_block is at B[r_b + j, c_b + k]
        C_block is at C[r_a + i, c_b + k]
        
        So:
        U index: r_a + i
        V index: (c_a + j) * p + (c_b + k) ? No.
        V is indexed by [j * p + k] in the seed.
        Seed: V[j * p + k, c].
        Here j is the middle index (col of A, row of B).
        k is the output col index.
        
        So V index: (c_a + j) * p + (c_b + k)
        W index: (c_b + k) * n + (r_a + i)
        """
        nonlocal current_rank
        # Each term in the decomposition corresponds to one column in U, V, W
        # For a rank-1 term with coefficient 1:
        # U[u_idx][current_rank] += coeff_u
        # V[v_idx][current_rank] += coeff_v
        # W[w_idx][current_rank] += coeff_w
        
        # We need to ensure U, V, W have enough rows.
        # We'll append to a list of dicts or just extend.
        
        pass # Placeholder, logic below
    
    # To make it efficient, we'll store U, V, W as lists of lists, growing them.
    U = [0] * (n * m)
    V = [0] * (m * p)
    W = [0] * (p * n)
    # Wait, U, V, W are matrices.
    # Let's use lists of lists where we append a new column (rank dim) for each term.
    
    U_cols = [] # Each element is a dict or sparse representation? No, just dense lists is fine for small rank.
    # Rank ~ 80. n*m = 12. m*p = 28. p*n = 21.
    # 80 * 12 = 960 entries. Small.
    
    U_mat = [[0] * 80 for _ in range(n * m)] # Pre-allocate max possible rank? No, better to append.
    V_mat = [[0] * 80 for _ in range(m * p)]
    W_mat = [[0] * 80 for _ in range(p * n)]
    
    rank = 0
    
    def add_rank1_term(u_idxs, v_idxs, w_idxs, coeff=1):
        nonlocal rank
        if rank >= len(U_mat[0]):
             # Extend
             for r in range(len(U_mat)): U_mat[r].append(0)
             for r in range(len(V_mat)): V_mat[r].append(0)
             for r in range(len(W_mat)): W_mat[r].append(0)
        
        for idx in u_idxs:
            U_mat[idx][rank] += coeff
        for idx in v_idxs:
            V_mat[idx][rank] += coeff
        for idx in w_idxs:
            W_mat[idx][rank] += coeff
        rank += 1
        
    def add_strassen_2x2x2(A_r0, A_c0, B_r0, B_c0):
        """
        Adds rank-7 Strassen terms for A[A_r0:A_r0+2, A_c0:A_c0+2] * B[B_r0:B_r0+2, B_c0:B_c0+2]
        """
        # Local indices 0,1 map to global:
        # A: (A_r0+0, A_c0+0), (A_r0+0, A_c0+1), (A_r0+1, A_c0+0), (A_r0+1, A_c0+1)
        # B: (B_r0+0, B_c0+0), ...
        
        # U indices for A elements:
        u00 = A_r0 * m + A_c0
        u01 = A_r0 * m + A_c0 + 1
        u10 = (A_r0 + 1) * m + A_c0
        u11 = (A_r0 + 1) * m + A_c0 + 1
        
        # V indices for B elements:
        # V index = j * p + k
        # j is row of B, k is col of B
        v00 = B_r0 * p + B_c0
        v01 = B_r0 * p + B_c0 + 1
        v10 = (B_r0 + 1) * p + B_c0
        v11 = (B_r0 + 1) * p + B_c0 + 1
        
        # W indices for C elements:
        # W index = k * n + i
        # k is col of C (which is col of B), i is row of C (which is row of A)
        w00 = B_c0 * n + A_r0
        w01 = (B_c0 + 1) * n + A_r0
        w10 = B_c0 * n + (A_r0 + 1)
        w11 = (B_c0 + 1) * n + (A_r0 + 1)
        
        # Strassen terms:
        # 1. (A00 + A11)(B00 + B11)
        add_rank1_term([u00, u11], [v00, v11], [w00, w11], 1)
        
        # 2. (A10 + A11)B00
        add_rank1_term([u10, u11], [v00], [w10, w11], 1)
        
        # 3. A00(B01 - B11)
        add_rank1_term([u00], [v01], [w01], 1)
        add_rank1_term([u00], [v11], [w11], -1)
        
        # 4. A11(B10 - B00)
        add_rank1_term([u11], [v10], [w10], 1)
        add_rank1_term([u11], [v00], [w00], -1)
        
        # 5. (A00 + A01)B11
        add_rank1_term([u00, u01], [v11], [w10, w11], 1)
        
        # 6. (A10 - A00)(B00 + B01)
        add_rank1_term([u10], [v00, v01], [w10, w11], 1)
        add_rank1_term([u00], [v00, v01], [w00, w01], -1)
        
        # 7. (A01 - A11)(B10 + B11)
        add_rank1_term([u01], [v10, v11], [w00, w10], 1) # Check W indices for this term?
        # C01 = M3 + M5. C10 = M2 + M4.
        # Let's verify Strassen W indices.
        # T[i,j,k] = sum U[i,r] V[j,r] W[k,r]
        # Term 7: (A01 - A11)(B10 + B11)
        # U: A01 (+), A11 (-)
        # V: B10 (+), B11 (+)
        # W: This product contributes to C00? No.
        # C00 = M1 - M4 + M7 - M5
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M6 - M2
        
        # M7 contributes to C00 and C11?
        # M7 = (A01 - A11)(B10 + B11)
        # In standard Strassen:
        # C00 = M1 - M4 + M7 - M5
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M6 - M2
        
        # So M7 is added to C00 with +1 and to C11 with +1?
        # Wait, standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M4 + M7 - M5
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M6 - M2
        
        # So M7 contributes to C00 (coeff +1) and C11 (coeff +1)?
        # No, M7 is just one term.
        # The tensor decomposition sums over r.
        # For term r=7:
        # U[i,7] V[j,7] W[k,7]
        # We need W[k,7] to be 1 for k where C[k] gets M7.
        # C00 gets M7. So W[0,7] = 1.
        # C11 gets M7? No, C11 = M1 - M3 + M6 - M2.
        # C00 = M1 - M4 + M7 - M5.
        # So only C00 gets M7?
        # Let's re-verify Strassen C formulas.
        # C00 = M1 - M4 + M7 - M5
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M6 - M2
        
        # Yes, M7 only appears in C00.
        # Wait, is that right?
        # Let's check a source.
        # Wikipedia:
        # C00 = M1 - M4 + M7 - M5
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M6 - M2
        
        # So M7 is ONLY in C00.
        # Therefore, for term 7, W index should only be w00 (local 0,0).
        
        add_rank1_term([u01], [v10, v11], [w00], 1)
        add_rank1_term([u11], [v10, v11], [w00], -1)
        
        # Re-check term 3: A00(B01-B11). Contributes to C01.
        # C01 = M3 + M5.
        # So W index should be w01.
        # My code: add_rank1_term([u00], [v01], [w01], 1) -> Correct.
        # add_rank1_term([u00], [v11], [w11], -1) -> WRONG. W index should be w01 for both?
        # No, V indices are different.
        # Term 3: U=[u00], V=[v01, v11], W=[w01]
        # We split V into two terms?
        # No, a rank-1 term has one V vector.
        # If V is [0, 1] (local), it's one vector.
        # So Term 3 is:
        # U: [u00]
        # V: [v01, v11] (with coeffs 1, -1)
        # W: [w01]
        # This is ONE rank-1 term?
        # No, U, V, W are vectors.
        # U = e_u00
        # V = e_v01 - e_v11
        # W = e_w01
        # This is a valid rank-1 term in the tensor space.
        
        # So I need to modify add_rank1_term to handle vectors with coefficients,
        # or just add multiple scalar rank-1 terms which sums to the same.
        # Adding 2 scalar rank-1 terms increases rank by 2.
        # Using a vector with 2 non-zero entries increases rank by 1.
        # So I MUST use the vector representation.
        
    # Redefine add_rank1_term to take lists of (index, coeff)
    
    # Reset
    U_mat = [[] for _ in range(n * m)]
    V_mat = [[] for _ in range(m * p)]
    W_mat = [[] for _ in range(p * n)]
    rank = 0
    
    def add_term(u_pairs, v_pairs, w_pairs):
        nonlocal rank
        # Extend
        for r in range(len(U_mat)): U_mat[r].append(0)
        for r in range(len(V_mat)): V_mat[r].append(0)
        for r in range(len(W_mat)): W_mat[r].append(0)
        
        for idx, coeff in u_pairs:
            U_mat[idx][rank] += coeff
        for idx, coeff in v_pairs:
            V_mat[idx][rank] += coeff
        for idx, coeff in w_pairs:
            W_mat[idx][rank] += coeff
        rank += 1
        
    def add_strassen_2x2x2(A_r0, A_c0, B_r0, B_c0):
        u00 = A_r0 * m + A_c0
        u01 = A_r0 * m + A_c0 + 1
        u10 = (A_r0 + 1) * m + A_c0
        u11 = (A_r0 + 1) * m + A_c0 + 1
        
        v00 = B_r0 * p + B_c0
        v01 = B_r0 * p + B_c0 + 1
        v10 = (B_r0 + 1) * p + B_c0
        v11 = (B_r0 + 1) * p + B_c0 + 1
        
        w00 = B_c0 * n + A_r0
        w01 = (B_c0 + 1) * n + A_r0
        w10 = B_c0 * n + (A_r0 + 1)
        w11 = (B_c0 + 1) * n + (A_r0 + 1)
        
        # 1. M1
        add_term([(u00,1), (u11,1)], [(v00,1), (v11,1)], [(w00,1), (w11,1)])
        
        # 2. M2
        add_term([(u10,1), (u11,1)], [(v00,1)], [(w10,1), (w11,1)])
        
        # 3. M3
        add_term([(u00,1)], [(v01,1), (v11,-1)], [(w01,1)])
        
        # 4. M4
        add_term([(u11,1)], [(v10,1), (v00,-1)], [(w10,1)])
        
        # 5. M5
        add_term([(u00,1), (u01,1)], [(v11,1)], [(w10,1), (w11,1)])
        
        # 6. M6
        add_term([(u10,1), (u00,-1)], [(v00,1), (v01,1)], [(w10,1), (w11,1)])
        
        # 7. M7
        add_term([(u01,1), (u11,-1)], [(v10,1), (v11,1)], [(w00,1)])
        
    def add_schoolbook_ar_bk(A_r0, A_c0, A_h, A_w, B_r0, B_c0, B_h, B_w):
        # A_h x A_w * B_h x B_w
        # A is at A_r0, A_c0
        # B is at B_r0, B_c0
        for i in range(A_h):
            for j in range(A_w):
                for k in range(B_w):
                    u_idx = (A_r0 + i) * m + (A_c0 + j)
                    v_idx = (A_c0 + j) * p + (B_c0 + k)
                    w_idx = (B_c0 + k) * n + (A_r0 + i)
                    add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])
                    
    # Apply blocks
    # 1. A11 (2x2) * B11 (2x2) -> Strassen
    add_strassen_2x2x2(0, 0, 0, 0)
    
    # 2. A11 (2x2) * B12 (2x2) -> Strassen
    add_strassen_2x2x2(0, 0, 0, 2)
    
    # 3. A11 (2x2) * B13 (2x3) -> Schoolbook
    add_schoolbook_ar_bk(0, 0, 2, 2, 0, 4, 2, 3)
    
    # 4. A12 (2x2) * B21 (2x2) -> Strassen
    add_strassen_2x2x2(0, 2, 2, 0)
    
    # 5. A12 (2x2) * B22 (2x2) -> Strassen
    add_strassen_2x2x2(0, 2, 2, 2)
    
    # 6. A12 (2x2) * B23 (2x3) -> Schoolbook
    add_schoolbook_ar_bk(0, 2, 2, 2, 2, 4, 2, 3)
    
    # 7. A21 (1x2) * B11 (2x2) -> Schoolbook
    add_schoolbook_ar_bk(2, 0, 1, 2, 0, 0, 2, 2)
    
    # 8. A21 (1x2) * B12 (2x2) -> Schoolbook
    add_schoolbook_ar_bk(2, 0, 1, 2, 0, 2, 2, 2)
    
    # 9. A21 (1x2) * B13 (2x3) -> Schoolbook
    add_schoolbook_ar_bk(2, 0, 1, 2, 0, 4, 2, 3)
    
    # 10. A22 (1x2) * B21 (2x2) -> Schoolbook
    add_schoolbook_ar_bk(2, 2, 1, 2, 2, 0, 2, 2)
    
    # 11. A22 (1x2) * B22 (2x2) -> Schoolbook
    add_schoolbook_ar_bk(2, 2, 1, 2, 2, 2, 2, 2)
    
    # 12. A22 (1x2) * B23 (2x3) -> Schoolbook
    add_schoolbook_ar_bk(2, 2, 1, 2, 2, 4, 2, 3)
    
    # Finalize
    U = U_mat
    V = V_mat
    W = W_mat
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
