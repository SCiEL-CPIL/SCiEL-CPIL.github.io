"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 6  # A is n x m, B is m x p, C is n x p

    # We will build the tensor decomposition for multiplying (n x m) * (m x p) -> (n x p)
    # using Strassen's algorithm for 2x2 blocks as a building block.
    #
    # Strategy:
    # - Partition A into 2x2 blocks of size 2x2 (so 2 block-rows, 2 block-cols)
    # - Partition B into 2x2 blocks of size 2x2 (so 2 block-rows, 3 block-cols since p=6)
    # - Partition C into 2x2 blocks of size 2x2 (so 2 block-rows, 3 block-cols)
    #
    # For each pair of block-row i and block-col k in C, we compute:
    #   C[i,k] = sum_j A[i,j] * B[j,k]  (sum over 2 block-rows j=0,1)
    #
    # Each block multiplication A[i,j] * B[j,k] is a 2x2 * 2x2 = 2x2 multiplication,
    # which can be done with rank 7 using Strassen.
    #
    # Total blocks: 2 (block-rows) * 2 (block-cols of A) * 3 (block-cols of B) = 12 block multiplications
    # Each with rank 7, total rank = 12 * 7 = 84.
    #
    # But we can do better: for each fixed block-row i and block-col k of C,
    # we need to compute sum over j=0,1 of A[i,j] * B[j,k].
    # This is a 2-term sum of 2x2*2x2 products.
    #
    # Actually, let's just use schoolbook for each 2x2*2x2 block multiplication.
    # Schoolbook 2x2*2x2 has rank 4 (each of the 4 entries is one product).
    # Total: 12 blocks * 4 = 48.
    #
    # Even better: use Strassen for each 2x2*2x2, rank 7 each.
    # Total: 12 * 7 = 84. That's worse than schoolbook.
    #
    # Wait, schoolbook for the full 4x4x6 is rank 96 (4*4*6).
    # Using 2x2 blocks with schoolbook: 12 blocks * 4 = 48. Better!
    #
    # Let me verify: A is 4x4, so 2 block-rows (each 2 rows) and 2 block-cols (each 2 cols).
    # B is 4x6, so 2 block-rows (each 2 rows) and 3 block-cols (each 2 cols).
    # C is 4x6, so 2 block-rows and 3 block-cols.
    #
    # For C block (i,k), we sum over j=0,1: A block (i,j) * B block (j,k).
    # Each A block (i,j) is 2x2, each B block (j,k) is 2x2, product is 2x2.
    # Schoolbook: 4 multiplications per block product.
    # Two block products summed: 8 multiplications per C block.
    # 6 C blocks total: 6 * 8 = 48.
    #
    # So rank 48 with schoolbook on 2x2 blocks.
    #
    # Now let's implement this.

    # Block structure:
    # A: block_row i in {0,1}, block_col j in {0,1}, each block is 2x2
    #    A[i,j] has entries A[2*i + r, 2*j + c] for r,c in {0,1}
    # B: block_row j in {0,1}, block_col k in {0,1,2}, each block is 2x2
    #    B[j,k] has entries B[2*j + r, 2*k + c] for r,c in {0,1}
    # C: block_row i in {0,1}, block_col k in {0,1,2}, each block is 2x2
    #    C[i,k] has entries C[2*i + r, 2*k + c] for r,c in {0,1}

    # For each C block (i,k), compute sum_{j=0,1} A[i,j] * B[j,k]
    # Using schoolbook for each 2x2*2x2:
    #   C_block[r,c] = sum_{j=0,1} sum_{s=0,1} A[i,j][r,s] * B[j,k][s,c]

    # Each term A[i,j][r,s] * B[j,k][s,c] is a rank-1 term.
    # Total terms: for each (i,k) in 2x3, for each j in {0,1}, for each (r,s,c) in 2x2x2:
    # 6 * 2 * 8 = 96. That's the same as full schoolbook!

    # I made an error. Let me reconsider.
    #
    # The issue is that when we decompose into blocks, we're still doing the same number of
    # scalar multiplications. The block decomposition doesn't reduce the rank unless we
    # use a non-schoolbook algorithm for the block multiplications.
    #
    # Let's use Strassen for each 2x2*2x2 block multiplication.
    # Strassen: 7 multiplications per 2x2*2x2.
    # For each C block (i,k), we have 2 block products (j=0,1), each with 7 mults.
    # Total: 6 * 2 * 7 = 84. Still worse than 96? No, 84 < 96.
    #
    # Wait, I need to be more careful. When we sum two 2x2 products, we can't just
    # independently apply Strassen to each. We need to combine them.
    #
    # Actually, let's think differently. For each C block (i,k):
    #   C[i,k] = A[i,0] * B[0,k] + A[i,1] * B[1,k]
    #
    # This is a sum of two 2x2*2x2 products. We can view this as multiplying
    # a 2x4 matrix (A[i,0] | A[i,1]) by a 4x2 matrix (B[0,k]; B[1,k]) to get a 2x2.
    #
    # The 2x4 * 4x2 multiplication has rank at most 2*4*2 = 16 (schoolbook).
    # But we can do better.
    #
    # Actually, let's just use the full schoolbook and accept rank 96.
    # Or better, let me try a different approach.
    #
    # Alternative: use the known rank-18 decomposition for 4x4x4, and extend to 4x4x6.
    #
    # For 4x4x4, there's a known rank-18 decomposition. Let me try to find one.
    #
    # Actually, the best known upper bound for 4x4x4 is 18 (Alon et al., 1989).
    # Let me try to construct it using 2x2 Strassen blocks.
    #
    # 4x4x4 = 4 blocks of 2x2x2. If each 2x2x2 uses rank 7 (Strassen), total rank 28.
    # But we can share some multiplications across blocks to get down to 18.
    #
    # This is complex. Let me just use a verified correct decomposition.
    #
    # Simplest correct approach: schoolbook with rank 96.
    # But let me try to get a lower rank.
    #
    # Let me use the following approach:
    # - For the 4x4x4 part, use a rank-28 decomposition (4 Strassen blocks).
    # - For the 4x4x2 part, use schoolbook with rank 32.
    # - Total: 28 + 32 = 60.
    #
    # Actually, 4x4x2 can be done more efficiently.
    # 4x4x2: A is 4x4, B is 4x2, C is 4x2.
    # This is like computing 2 independent 4x4x1 multiplications.
    # Each 4x4x1 has rank at most 16 (schoolbook).
    # Total: 32.
    #
    # But we can do 4x4x2 with rank 24 using some optimization.
    #
    # For simplicity, let's use:
    # - 4x4x4 with rank 28 (4 Strassen blocks)
    # - 4x4x2 with rank 32 (schoolbook)
    # - Total: 60
    #
    # Let me implement the Strassen 2x2x2 block decomposition.

    # Strassen 2x2x2:
    # Given A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A * B = [[c00, c01], [c10, c11]]
    #
    # 7 products:
    # m1 = (a00 + a11) * (b00 + b11)
    # m2 = (a10 + a11) * b00
    # m3 = a00 * (b10 - b11)
    # m4 = a11 * (b01 - b00)
    # m5 = (a00 + a01) * b11
    # m6 = (a10 - a00) * (b00 + b01)
    # m7 = (a01 - a11) * (b10 + b11)
    #
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6

    # For each 2x2x2 block, we create 7 rank-1 terms.
    # Each term has a coefficient in U, V, W.

    def strassen_block(U, V, W, base_col, a_offset, b_offset, c_offset):
        """Add Strassen 2x2x2 decomposition for blocks at given offsets.
        
        a_offset: (row_offset, col_offset) for A block (2x2)
        b_offset: (row_offset, col_offset) for B block (2x2)  
        c_offset: (row_offset, col_offset) for C block (2x2)
        """
        a_r, a_c = a_offset
        b_r, b_c = b_offset
        c_r, c_c = c_offset
        
        # Local indices 0,1 for rows and cols within each block
        # A entries: A[a_r + r, a_c + c] for r,c in {0,1}
        # B entries: B[b_r + r, b_c + c] for r,c in {0,1}
        # C entries: C[c_r + r, c_c + c] for r,c in {0,1}
        
        # m1 = (a00 + a11) * (b00 + b11)
        col = base_col
        U[a_r * m + a_c][col] += 1          # a00
        U[(a_r+1) * m + (a_c+1)][col] += 1  # a11
        V[b_r * p + b_c][col] += 1          # b00
        V[(b_r+1) * p + (b_c+1)][col] += 1  # b11
        W[c_r * n + c_c][col] += 1          # c00
        W[(c_r+1) * n + (c_c+1)][col] += 1  # c11
        col += 1
        
        # m2 = (a10 + a11) * b00
        U[(a_r+1) * m + a_c][col] += 1      # a10
        U[(a_r+1) * m + (a_c+1)][col] += 1  # a11
        V[b_r * p + b_c][col] += 1          # b00
        W[(c_r+1) * n + c_c][col] += 1      # c10
        col += 1
        
        # m3 = a00 * (b10 - b11)
        U[a_r * m + a_c][col] += 1          # a00
        V[(b_r+1) * p + b_c][col] += 1      # b10
        V[(b_r+1) * p + (b_c+1)][col] -= 1  # -b11
        W[c_r * n + (c_c+1)][col] += 1      # c01
        col += 1
        
        # m4 = a11 * (b01 - b00)
        U[(a_r+1) * m + (a_c+1)][col] += 1  # a11
        V[b_r * p + (b_c+1)][col] += 1      # b01
        V[b_r * p + b_c][col] -= 1          # -b00
        W[c_r * n + c_c][col] += 1          # c00
        W[(c_r+1) * n + (c_c+1)][col] += 1  # c11
        col += 1
        
        # m5 = (a00 + a01) * b11
        U[a_r * m + a_c][col] += 1          # a00
        U[a_r * m + (a_c+1)][col] += 1      # a01
        V[(b_r+1) * p + (b_c+1)][col] += 1  # b11
        W[c_r * n + c_c][col] -= 1          # -c00
        W[c_r * n + (c_c+1)][col] += 1      # c01
        col += 1
        
        # m6 = (a10 - a00) * (b00 + b01)
        U[(a_r+1) * m + a_c][col] += 1      # a10
        U[a_r * m + a_c][col] -= 1          # -a00
        V[b_r * p + b_c][col] += 1          # b00
        V[b_r * p + (b_c+1)][col] += 1      # b01
        W[(c_r+1) * n + (c_c+1)][col] += 1  # c11
        col += 1
        
        # m7 = (a01 - a11) * (b10 + b11)
        U[a_r * m + (a_c+1)][col] += 1      # a01
        U[(a_r+1) * m + (a_c+1)][col] -= 1  # -a11
        V[(b_r+1) * p + b_c][col] += 1      # b10
        V[(b_r+1) * p + (b_c+1)][col] += 1  # b11
        W[c_r * n + c_c][col] += 1          # c00
        W[(c_r+1) * n + c_c][col] += 1      # c10
        col += 1
        
        return col

    # Now build the full decomposition.
    # A is 4x4: 2x2 blocks of 2x2
    # B is 4x6: 2x3 blocks of 2x2
    # C is 4x6: 2x3 blocks of 2x2
    #
    # For each C block (i,k) where i in {0,1}, k in {0,1,2}:
    #   C[i,k] = A[i,0] * B[0,k] + A[i,1] * B[1,k]
    #
    # We can't directly apply Strassen to a sum of two products.
    # Instead, let's use schoolbook for each 2x2*2x2 and accept rank 96.
    #
    # Or, let's use a different strategy: treat the full 4x4x6 as a single tensor
    # and use a known decomposition.
    #
    # For now, let's use schoolbook with rank 96 to ensure correctness.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
