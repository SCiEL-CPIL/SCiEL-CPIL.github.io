"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 1. Define the base Strassen decomposition for 2x2x2
    # U[4, 7], V[4, 7], W[4, 7]
    # Indices: 0:11, 1:12, 2:21, 3:22
    U_base = np.zeros((4, 7), dtype=int)
    V_base = np.zeros((4, 7), dtype=int)
    W_base = np.zeros((4, 7), dtype=int)
    
    # M1 = A11 * B11
    U_base[0, 0] = 1
    V_base[0, 0] = 1
    W_base[0, 0] = 1
    W_base[2, 0] = 0 # W[22] doesn't get M1 directly, W[11] does.
    # Correction: W[k*n+i, c].
    # For 2x2:
    # C11 = A11*B11 + ...
    # C12 = A11*B12 + ...
    # C21 = A21*B11 + ...
    # C22 = A21*B12 + ...
    
    # Strassen Steps:
    # M1 = A11 * B11
    # M2 = (A11 + A21) * B12
    # M3 = (A12 + A22) * B21
    # M4 = A22 * B22
    # M5 = (A11 + A22) * (B11 + B22)
    # M6 = (A12 - A22) * (B21 + B22)
    # M7 = (A11 - A21) * (B11 + B12)
    
    # C11 = M1 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M4 + M6
    
    # U (for A)
    # Col 0 (M1): A11
    U_base[0, 0] = 1
    # Col 1 (M2): A11 + A21
    U_base[0, 1] = 1
    U_base[2, 1] = 1
    # Col 2 (M3): A12 + A22
    U_base[1, 2] = 1
    U_base[3, 2] = 1
    # Col 3 (M4): A22
    U_base[3, 3] = 1
    # Col 4 (M5): A11 + A22
    U_base[0, 4] = 1
    U_base[3, 4] = 1
    # Col 5 (M6): A12 - A22
    U_base[1, 5] = 1
    U_base[3, 5] = -1
    # Col 6 (M7): A11 - A21
    U_base[0, 6] = 1
    U_base[2, 6] = -1
    
    # V (for B)
    # Col 0 (M1): B11
    V_base[0, 0] = 1
    # Col 1 (M2): B12
    V_base[1, 1] = 1
    # Col 2 (M3): B21
    V_base[2, 2] = 1
    # Col 3 (M4): B22
    V_base[3, 3] = 1
    # Col 4 (M5): B11 + B22
    V_base[0, 4] = 1
    V_base[3, 4] = 1
    # Col 5 (M6): B21 + B22
    V_base[2, 5] = 1
    V_base[3, 5] = 1
    # Col 6 (M7): B11 + B12
    V_base[0, 6] = 1
    V_base[1, 6] = 1
    
    # W (for C)
    # C11 = M1 - M5 + M7
    W_base[0, 0] = 1
    W_base[0, 4] = -1
    W_base[0, 6] = 1
    # C12 = M3 + M5
    W_base[1, 2] = 1
    W_base[1, 4] = 1
    # C21 = M2 + M4
    W_base[2, 1] = 1
    W_base[2, 3] = 1
    # C22 = M1 - M4 + M6
    W_base[3, 0] = 1
    W_base[3, 3] = -1
    W_base[3, 5] = 1

    def add_decomp(U_global, V_global, W_global, U_loc, V_loc, W_loc, i_off, j_off, k_off):
        """
        Adds a local decomposition to global matrices.
        i_off, j_off: row/col offsets for A (and C rows)
        k_off: col offset for B (and C cols)
        """
        rank_loc = U_loc.shape[1]
        for r in range(rank_loc):
            for i in range(U_loc.shape[0]):
                if U_loc[i, r] != 0:
                    U_global[i + i_off, r + k_off] += U_loc[i, r]
            for j in range(V_loc.shape[0]):
                if V_loc[j, r] != 0:
                    V_global[j + j_off, r + k_off] += V_loc[j, r]
            for k in range(W_loc.shape[0]):
                if W_loc[k, r] != 0:
                    W_global[k + k_off, r + k_off] += W_loc[k, r]

    # 2. Construct 4x4x4 decomposition (Rank 49)
    # Split into 4 blocks of 2x2
    # A: (0,0), (0,1), (1,0), (1,1)
    # B: (0,0), (0,1), (1,0), (1,1)
    # C: (0,0), (0,1), (1,0), (1,1)
    
    # We use the same Strassen logic on the block level.
    # But since the blocks are 2x2, we just map the base Strassen indices.
    
    U_444 = np.zeros((16, 49), dtype=int)
    V_444 = np.zeros((16, 49), dtype=int)
    W_444 = np.zeros((16, 49), dtype=int)
    
    # The 7 Strassen combinations map to:
    # M1: A00 * B00
    # M2: (A00+A10) * B01
    # M3: (A01+A11) * B10
    # M4: A11 * B11
    # M5: (A00+A11) * (B00+B11)
    # M6: (A01-A11) * (B10+B11)
    # M7: (A00-A10) * (B00+B01)
    
    # C00 = M1 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M4 + M6
    
    # We will add the base Strassen for each of the 7 multiplications.
    # Multiplication r (0..6) uses blocks:
    # U_blocks: which A blocks are involved and with what sign
    # V_blocks: which B blocks are involved and with what sign
    
    # M1: A00, B00
    add_decomp(U_444, V_444, W_444, U_base, V_base, W_base, 0, 0, 0) # i=0, j=0, k=0? 
    # Wait, add_decomp assumes U_loc, V_loc, W_loc are for a 2x2x2 multiplication.
    # The mapping of local indices (0..3) to global blocks:
    # A block (0,0) is rows 0-1, cols 0-1.
    # A block (0,1) is rows 0-1, cols 2-3.
    # A block (1,0) is rows 2-3, cols 0-1.
    # A block (1,1) is rows 2-3, cols 2-3.
    
    # So for M1 (A00 * B00):
    # A00 is at i=0, j=0.
    # B00 is at j=0, k=0 (k is col index of B, so B cols 0-1).
    # C00 is at k=0, i=0.
    # This matches the offsets i=0, j=0, k=0.
    
    # M2: (A00 + A10) * B01
    # A00 is (0,0), A10 is (2,0).
    # B01 is (0,2).
    # Result adds to C01 (0,2).
    # We need to add two decompositions:
    # 1. A00 * B01
    # 2. A10 * B01
    
    # Let's create a helper to add a weighted base Strassen.
    def add_weighted(U_g, V_g, W_g, i_off, j_off, k_off, sign=1):
        # This adds sign * (U_base[i, r], V_base[j, r], W_base[k, r])
        # But U_base, V_base, W_base have non-zero entries in specific local spots.
        # We just call add_decomp with scaled matrices?
        # Easier: iterate over base non-zeros.
        pass # We'll implement the 7 calls explicitly.

    # M1: A00 * B00 -> C00
    add_decomp(U_444, V_444, W_444, U_base, V_base, W_base, 0, 0, 0)
    
    # M2: (A00 + A10) * B01 -> C01
    # Term 1: A00 * B01
    add_decomp(U_444, V_444, W_444, U_base, V_base, W_base, 0, 2, 2) # i=0, j=2, k=2
    # Term 2: A10 * B01
    add_decomp(U_444, V_444, W_444, U_base, V_base, W_base, 2, 2, 2) # i=2, j=2, k=2
    
    # M3: (A01 + A11) * B10 -> C01
    # Term 1: A01 * B10
    add_decomp(U_444, V_444, W_444, U_base, V_base, W_base, 0, 0, 2) # i=0, j=0, k=2
    # Term 2: A11 * B10
    add_decomp(U_444, V_444, W_444, U_base, V_base, W_base, 2, 0, 2) # i=2, j=0, k=2
    
    # M4: A11 * B11 -> C11
    add_decomp(U_444, V_444, W_444, U_base, V_base, W_base, 2, 2, 4) # i=2, j=2, k=4
    
    # M5: (A00 + A11) * (B00 + B11) -> C00, C01, C10, C11 (distributed)
    # This one is tricky. The result of M5 is added to multiple C blocks with signs.
    # C00 gets -M5, C01 gets +M5, C10 gets +M5, C11 gets -M5.
    # We need to add 4 decompositions for A00*B00, A00*B11, A11*B00, A11*B11?
    # No, M5 is (A00+A11)*(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # We add these 4 terms, and then subtract/add the total result from the C blocks.
    
    # Term 1: A00 * B00
    add_decomp(U_444, V_444, W_444, U_base, V_base, W_base, 0, 0, 0)
    # Term 2: A00 * B11
    add_decomp(U_444, V_444, W_444, U_base, V_base, W_base, 0, 2, 0)
    # Term 3: A11 * B00
    add_decomp(U_444, V_444, W_444, U_base, V_base, W_base, 2, 0, 2)
    # Term 4: A11 * B11
    add_decomp(U_444, V_444, W_444, U_base, V_base, W_base, 2, 2, 2)
    
    # Now adjust W for M5 contributions.
    # We added the raw products. We need to subtract M5 from C00 and C11, and add to C01 and C10?
    # Wait, Strassen formula:
    # C00 = M1 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M4 + M6
    
    # My add_decomp calls above for M5 added the products to C00, C01, C10, C11?
    # No, add_decomp(i_off, j_off, k_off) adds to C block at (k_off, i_off).
    # For Term 1 (A00*B00), I used i=0, j=0, k=0. This adds to C00.
    # For Term 2 (A00*B11), I used i=0, j=2, k=0. This adds to C00?
    # B11 is cols 2-3. So k_off for B is 2?
    # My add_decomp signature: i_off (A row), j_off (A col / B row), k_off (B col / C col).
    # So B11 is j=2, k=2.
    # A00 is i=0, j=0.
    # So A00 * B11 should use i=0, j=2 (B row), k=2 (B col).
    # Result C is at i=0, k=2 -> C01.
    
    # I made mistakes in the offsets above. Let's restart the 444 construction carefully.

    U_444 = np.zeros((16, 49), dtype=int)
    V_444 = np.zeros((16, 49), dtype=int)
    W_444 = np.zeros((16, 49), dtype=int)
    
    def add_strassen_block(U_g, V_g, W_g, i_a, j_a, j_b, k_b, i_c, k_c, sign=1):
        # Adds sign * (A[i_a, j_a] * B[j_b, k_b]) to C[i_c, k_c]
        # i_a, j_a: A block indices (0/1 for row/col of 2x2 block)
        # j_b, k_b: B block indices
        # i_c, k_c: C block indices
        # Global offsets:
        # A: i_g = 2*i_a, j_g = 2*j_a
        # B: j_g = 2*j_b, k_g = 2*k_b
        # C: i_g = 2*i_c, k_g = 2*k_c
        for r in range(7):
            for u_r in range(4):
                val = U_base[u_r, r]
                if val != 0:
                    U_g[2*i_a + u_r, r] += sign * val
            for v_r in range(4):
                val = V_base[v_r, r]
                if val != 0:
                    V_g[2*j_b + v_r, r] += sign * val
            for w_r in range(4):
                val = W_base[w_r, r]
                if val != 0:
                    W_g[2*i_c + w_r, r] += sign * val

    # M1: A00 * B00 -> C00
    add_strassen_block(U_444, V_444, W_444, 0, 0, 0, 0, 0, 0, 1)
    # M1 also contributes to C11 with -1
    add_strassen_block(U_444, V_444, W_444, 0, 0, 0, 0, 1, 1, -1)
    
    # M2: (A00 + A10) * B01 -> C10
    # A00 * B01
    add_strassen_block(U_444, V_444, W_444, 0, 0, 0, 1, 1, 0, 1)
    # A10 * B01
    add_strassen_block(U_444, V_444, W_444, 1, 0, 0, 1, 1, 0, 1)
    
    # M3: (A01 + A11) * B10 -> C01
    # A01 * B10
    add_strassen_block(U_444, V_444, W_444, 0, 1, 1, 0, 0, 1, 1)
    # A11 * B10
    add_strassen_block(U_444, V_444, W_444, 1, 1, 1, 0, 0, 1, 1)
    
    # M4: A11 * B11 -> C10
    add_strassen_block(U_444, V_444, W_444, 1, 1, 1, 1, 1, 0, 1)
    # M4 also contributes to C11 with -1
    add_strassen_block(U_444, V_444, W_444, 1, 1, 1, 1, 1, 1, -1)
    
    # M5: (A00 + A11) * (B00 + B11)
    # Terms: A00B00, A00B11, A11B00, A11B11
    # Contributions: -C00, +C01, +C10, -C11
    
    # A00 * B00
    add_strassen_block(U_444, V_444, W_444, 0, 0, 0, 0, 0, 0, -1)
    add_strassen_block(U_444, V_444, W_444, 0, 0, 0, 0, 0, 1, 1)
    add_strassen_block(U_444, V_444, W_444, 0, 0, 0, 0, 1, 0, 1)
    add_strassen_block(U_444, V_444, W_444, 0, 0, 0, 0, 1, 1, -1)
    
    # A00 * B11
    add_strassen_block(U_444, V_444, W_444, 0, 0, 1, 1, 0, 0, -1)
    add_strassen_block(U_444, V_444, W_444, 0, 0, 1, 1, 0, 1, 1)
    add_strassen_block(U_444, V_444, W_444, 0, 0, 1, 1, 1, 0, 1)
    add_strassen_block(U_444, V_444, W_444, 0, 0, 1, 1, 1, 1, -1)
    
    # A11 * B00
    add_strassen_block(U_444, V_444, W_444, 1, 1, 0, 0, 0, 0, -1)
    add_strassen_block(U_444, V_444, W_444, 1, 1, 0, 0, 0, 1, 1)
    add_strassen_block(U_444, V_444, W_444, 1, 1, 0, 0, 1, 0, 1)
    add_strassen_block(U_444, V_444, W_444, 1, 1, 0, 0, 1, 1, -1)
    
    # A11 * B11
    add_strassen_block(U_444, V_444, W_444, 1, 1, 1, 1, 0, 0, -1)
    add_strassen_block(U_444, V_444, W_444, 1, 1, 1, 1, 0, 1, 1)
    add_strassen_block(U_444, V_444, W_444, 1, 1, 1, 1, 1, 0, 1)
    add_strassen_block(U_444, V_444, W_444, 1, 1, 1, 1, 1, 1, -1)
    
    # M6: (A01 - A11) * (B10 + B11) -> C11
    # Terms: A01B10, A01B11, -A11B10, -A11B11
    # Contribution: +C11
    
    # A01 * B10
    add_strassen_block(U_444, V_444, W_444, 0, 1, 1, 0, 1, 1, 1)
    # A01 * B11
    add_strassen_block(U_444, V_444, W_444, 0, 1, 1, 1, 1, 1, 1)
    # -A11 * B10
    add_strassen_block(U_444, V_444, W_444, 1, 1, 1, 0, 1, 1, -1)
    # -A11 * B11
    add_strassen_block(U_444, V_444, W_444, 1, 1, 1, 1, 1, 1, -1)
    
    # M7: (A00 - A10) * (B00 + B01) -> C00
    # Terms: A00B00, A00B01, -A10B00, -A10B01
    # Contribution: +C00
    
    # A00 * B00
    add_strassen_block(U_444, V_444, W_444, 0, 0, 0, 0, 0, 0, 1)
    # A00 * B01
    add_strassen_block(U_444, V_444, W_444, 0, 0, 0, 1, 0, 0, 1)
    # -A10 * B00
    add_strassen_block(U_444, V_444, W_444, 1, 0, 0, 0, 0, 0, -1)
    # -A10 * B01
    add_strassen_block(U_444, V_444, W_444, 1, 0, 0, 1, 0, 0, -1)

    # 3. Construct 4x4x2 decomposition (Rank 28)
    # A: 4 blocks. B: 2 blocks (B0, B1). C: 2 blocks (C0, C1).
    # C0 = A00B0 + A01B1
    # C1 = A10B0 + A11B1
    
    U_442 = np.zeros((16, 28), dtype=int)
    V_442 = np.zeros((8, 28), dtype=int)
    W_442 = np.zeros((8, 28), dtype=int)
    
    # A00 * B0 -> C0
    add_strassen_block(U_442, V_442, W_442, 0, 0, 0, 0, 0, 0, 1)
    # A01 * B1 -> C0
    # B1 is j=0, k=1 (cols 2-3 of B? No, B is 4x2. B0 is cols 0-1, B1 is cols 2-3? No, B is 4x2.
    # Wait, B is 4x2. Blocks are 2x2.
    # B0: rows 0-1, cols 0-1.
    # B1: rows 2-3, cols 0-1.
    # So B1 is j=1, k=0.
    # A01 is i=0, j=1.
    add_strassen_block(U_442, V_442, W_442, 0, 1, 1, 0, 0, 0, 1)
    
    # A10 * B0 -> C1
    add_strassen_block(U_442, V_442, W_442, 1, 0, 0, 0, 1, 0, 1)
    # A11 * B1 -> C1
    add_strassen_block(U_442, V_442, W_442, 1, 1, 1, 0, 1, 0, 1)

    # 4. Combine 4x4x4 and 4x4x2
    # Total Rank = 49 + 28 = 77
    rank = 77
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((24, rank), dtype=int)
    W = np.zeros((24, rank), dtype=int)
    
    # Copy 444 part
    U[:, :49] = U_444
    V[:, :49] = V_444
    W[:, :49] = W_444
    
    # Copy 442 part
    # V for 442 is 8x28. V for 444 is 16x49.
    # B in 444 is 4x4. B in 442 is 4x2.
    # B_total is 4x6.
    # V_total rows: 24.
    # V_444 uses rows 0-15 (for 4x4 B).
    # V_442 uses rows 0-7 (for 4x2 B).
    # We map 442 B to cols 4-5 of B_total?
    # No, 444 uses B cols 0-3. 442 uses B cols 4-5?
    # B_total is 4x6.
    # B_444 is first 4 cols.
    # B_442 is last 2 cols.
    # So V_442 should be placed in V[:, 16:24]?
    # V_442 has 8 rows (2 blocks of 2x2).
    # B_442 is 4x2. Rows 0-3.
    # So V_442 rows 0-7 map to V rows 16-23?
    # V_442 row index j (0..3) maps to V row 16 + j?
    # Wait, V_442 is constructed for B of size 4x2.
    # Rows 0-1 correspond to B rows 0-1.
    # Rows 2-3 correspond to B rows 2-3.
    # So V_442[0:2] -> V[16:18]
    # V_442[2:4] -> V[18:20]
    # V_442[4:6] -> V[20:22] (Wait, V_442 is 8x28? No, V_442 is for B 4x2.
    # My add_strassen_block for 442 used V_g of size 8x28?
    # No, B is 4x2. Blocks are 2x2.
    # V_442 should be 4x28?
    # In add_strassen_block, V_g[2*j_b + v_r, r].
    # j_b is 0 or 1.
    # So V_g has 4 rows.
    # I initialized V_442 as 8x28. That's wrong. It should be 4x28.
    
    # Let's fix V_442 initialization.
    V_442 = np.zeros((4, 28), dtype=int)
    # Re-run 442 construction with correct V size
    U_442 = np.zeros((16, 28), dtype=int)
    W_442 = np.zeros((8, 28), dtype=int) # C is 4x2. 2 blocks of 2x2. 8 rows?
    # C is 4x2. Rows 0-3.
    # W_g[2*i_c + w_r, r]. i_c is 0 or 1.
    # So W_g has 4 rows.
    W_442 = np.zeros((4, 28), dtype=int)
    
    add_strassen_block(U_442, V_442, W_442, 0, 0, 0, 0, 0, 0, 1)
    add_strassen_block(U_442, V_442, W_442, 0, 1, 1, 0, 0, 0, 1)
    add_strassen_block(U_442, V_442, W_442, 1, 0, 0, 0, 1, 0, 1)
    add_strassen_block(U_442, V_442, W_442, 1, 1, 1, 0, 1, 0, 1)
    
    # Combine
    # U: 16x77
    # V: 24x77
    # W: 24x77
    
    U[:, 49:] = U_442
    # V_442 (4x28) maps to V rows 16-19?
    # B_442 is cols 4-5 of B_total.
    # V rows for B cols 4-5 are 16-19.
    V[16:20, 49:] = V_442
    # W_442 (4x28) maps to W rows 16-19?
    # C_442 is cols 4-5 of C_total.
    # W rows for C cols 4-5 are 16-19.
    W[16:20, 49:] = W_442
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
