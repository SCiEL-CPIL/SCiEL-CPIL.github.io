"""Matrix multiplication 4x4x9 with Strassen-based decomposition."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split the 9 columns of B into three groups: 4, 4, 1
    # Use Strassen (rank 28) for the 4x4x4 blocks and schoolbook (rank 16) for 4x4x1
    # Total rank = 28 + 28 + 16 = 72
    
    rank = 72
    U = [[0] * rank for _ in range(n * m)]  # 16 x 72
    V = [[0] * rank for _ in range(m * p)]  # 36 x 72
    W = [[0] * rank for _ in range(p * n)]  # 36 x 72
    
    col = 0
    
    def add_rank1(i, j, k, c, col_idx):
        """Add a single rank-1 term with coefficient c."""
        U[i * m + j][col_idx] += c
        V[j * p + k][col_idx] += c
        W[k * n + i][col_idx] += c
    
    def add_strassen_4x4(i0, j0, k0, start_col):
        """
        Add Strassen decomposition for a 4x4x4 block.
        i0, j0, k0: starting indices (row in A, col in A/row in B, col in B)
        Uses 2x2 blocks of 2x2 matrices.
        Returns the next column index.
        """
        c = start_col
        
        # Define 2x2 block offsets
        # A blocks: A00, A01, A10, A11 (each 2x2)
        # B blocks: B00, B01, B10, B11 (each 2x2)
        
        # Strassen 7 products on 2x2 blocks:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B11)
        # M4 = A11(B01 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # Each product is a 2x2 matrix product, which requires 4 scalar multiplications
        # Total: 7 * 4 = 28 rank-1 terms
        
        # Helper to get (i, j) for A block (block_row, block_col) at position (u, v)
        def a_idx(block_r, block_c, u, v):
            return i0 + block_r * 2 + u, j0 + block_c * 2 + v
        
        # Helper to get (j, k) for B block (block_row, block_col) at position (u, v)
        def b_idx(block_r, block_c, u, v):
            return j0 + block_r * 2 + u, k0 + block_c * 2 + v
        
        # Helper to add a 2x2 matrix product (schoolbook)
        # A_combos: list of (coeff, block_r, block_c, u, v) for A
        # B_combos: list of (coeff, block_r, block_c, u, v) for B
        def add_2x2_product(A_combos, B_combos, col_start):
            nonlocal c
            # For each of the 4 output positions (u, w):
            for u in range(2):
                for w in range(2):
                    # C[u][w] = sum_v A[u][v] * B[v][w]
                    for v in range(2):
                        # Expand A[u][v] and B[v][w]
                        for (ca, ar, ac, au, av) in A_combos:
                            if au == u and av == v:
                                for (cb, br, bc, bu, bv) in B_combos:
                                    if bu == v and bv == w:
                                        i = i0 + ar * 2 + au
                                        j = j0 + ac * 2 + av
                                        k = k0 + bc * 2 + bv
                                        # Wait, need to be careful with indices
                                        # A is at (i, j) where i = i0 + ar*2 + au, j = j0 + ac*2 + av
                                        # B is at (j, k) where j = j0 + br*2 + bu, k = k0 + bc*2 + bv
                                        # But we need j from A to equal j from B
                                        # Actually, the Strassen formulas already handle this
                                        # Let me reconsider the approach
                                        pass
            return c
        
        # Simpler approach: directly enumerate the 28 terms
        # M1 = (A00+A11)(B00+B11)
        # A00: rows i0..i0+1, cols j0..j0+1
        # A11: rows i0+2..i0+3, cols j0+2..j0+3
        # B00: rows j0..j0+1, cols k0..k0+1
        # B11: rows j0+2..j0+3, cols k0+2..k0+3
        
        # M1[u][w] = sum_v (A00[u][v] + A11[u][v]) * (B00[v][w] + B11[v][w])
        # = sum_v A00[u][v]*B00[v][w] + sum_v A00[u][v]*B11[v][w] + sum_v A11[u][v]*B00[v][w] + sum_v A11[u][v]*B11[v][w]
        
        # For each u, w, v:
        for u in range(2):
            for w in range(2):
                for v in range(2):
                    # A00[u][v] * B00[v][w]
                    i1 = i0 + u
                    j1 = j0 + v
                    add_rank1(i1, j1, k0 + w, 1, c); c += 1
                    
                    # A00[u][v] * B11[v][w]
                    i2 = i0 + u
                    j2 = j0 + v
                    add_rank1(i2, j2, k0 + 2 + w, 1, c); c += 1
                    
                    # A11[u][v] * B00[v][w]
                    i3 = i0 + 2 + u
                    j3 = j0 + 2 + v
                    add_rank1(i3, j3, k0 + w, 1, c); c += 1
                    
                    # A11[u][v] * B11[v][w]
                    i4 = i0 + 2 + u
                    j4 = j0 + 2 + v
                    add_rank1(i4, j4, k0 + 2 + w, 1, c); c += 1
        
        # M2 = (A10+A11)B00
        # A10: rows i0+2..i0+3, cols j0..j0+1
        # A11: rows i0+2..i0+3, cols j0+2..j0+3
        # B00: rows j0..j0+1, cols k0..k0+1
        for u in range(2):
            for w in range(2):
                for v in range(2):
                    # A10[u][v] * B00[v][w]
                    i1 = i0 + 2 + u
                    j1 = j0 + v
                    add_rank1(i1, j1, k0 + w, 1, c); c += 1
                    
                    # A11[u][v] * B00[v][w]
                    i2 = i0 + 2 + u
                    j2 = j0 + 2 + v
                    add_rank1(i2, j2, k0 + w, 1, c); c += 1
        
        # M3 = A00(B10-B11)
        # A00: rows i0..i0+1, cols j0..j0+1
        # B10: rows j0+2..j0+3, cols k0..k0+1
        # B11: rows j0+2..j0+3, cols k0+2..k0+3
        for u in range(2):
            for w in range(2):
                for v in range(2):
                    # A00[u][v] * B10[v][w]
                    i1 = i0 + u
                    j1 = j0 + v
                    add_rank1(i1, j1, k0 + w, 1, c); c += 1
                    
                    # A00[u][v] * (-B11[v][w])
                    i2 = i0 + u
                    j2 = j0 + v
                    add_rank1(i2, j2, k0 + 2 + w, -1, c); c += 1
        
        # M4 = A11(B01-B00)
        # A11: rows i0+2..i0+3, cols j0+2..j0+3
        # B01: rows j0..j0+1, cols k0+2..k0+3
        # B00: rows j0..j0+1, cols k0..k0+1
        for u in range(2):
            for w in range(2):
                for v in range(2):
                    # A11[u][v] * B01[v][w]
                    i1 = i0 + 2 + u
                    j1 = j0 + 2 + v
                    add_rank1(i1, j1, k0 + 2 + w, 1, c); c += 1
                    
                    # A11[u][v] * (-B00[v][w])
                    i2 = i0 + 2 + u
                    j2 = j0 + 2 + v
                    add_rank1(i2, j2, k0 + w, -1, c); c += 1
        
        # M5 = (A00+A01)B11
        # A00: rows i0..i0+1, cols j0..j0+1
        # A01: rows i0..i0+1, cols j0+2..j0+3
        # B11: rows j0+2..j0+3, cols k0+2..k0+3
        for u in range(2):
            for w in range(2):
                for v in range(2):
                    # A00[u][v] * B11[v][w]
                    i1 = i0 + u
                    j1 = j0 + v
                    add_rank1(i1, j1, k0 + 2 + w, 1, c); c += 1
                    
                    # A01[u][v] * B11[v][w]
                    i2 = i0 + u
                    j2 = j0 + 2 + v
                    add_rank1(i2, j2, k0 + 2 + w, 1, c); c += 1
        
        # M6 = (A10-A00)(B00+B01)
        # A10: rows i0+2..i0+3, cols j0..j0+1
        # A00: rows i0..i0+1, cols j0..j0+1
        # B00: rows j0..j0+1, cols k0..k0+1
        # B01: rows j0..j0+1, cols k0+2..k0+3
        for u in range(2):
            for w in range(2):
                for v in range(2):
                    # A10[u][v] * B00[v][w]
                    i1 = i0 + 2 + u
                    j1 = j0 + v
                    add_rank1(i1, j1, k0 + w, 1, c); c += 1
                    
                    # A10[u][v] * B01[v][w]
                    i2 = i0 + 2 + u
                    j2 = j0 + v
                    add_rank1(i2, j2, k0 + 2 + w, 1, c); c += 1
                    
                    # (-A00[u][v]) * B00[v][w]
                    i3 = i0 + u
                    j3 = j0 + v
                    add_rank1(i3, j3, k0 + w, -1, c); c += 1
                    
                    # (-A00[u][v]) * B01[v][w]
                    i4 = i0 + u
                    j4 = j0 + v
                    add_rank1(i4, j4, k0 + 2 + w, -1, c); c += 1
        
        # M7 = (A01-A11)(B10+B11)
        # A01: rows i0..i0+1, cols j0+2..j0+3
        # A11: rows i0+2..i0+3, cols j0+2..j0+3
        # B10: rows j0+2..j0+3, cols k0..k0+1
        # B11: rows j0+2..j0+3, cols k0+2..k0+3
        for u in range(2):
            for w in range(2):
                for v in range(2):
                    # A01[u][v] * B10[v][w]
                    i1 = i0 + u
                    j1 = j0 + 2 + v
                    add_rank1(i1, j1, k0 + w, 1, c); c += 1
                    
                    # A01[u][v] * B11[v][w]
                    i2 = i0 + u
                    j2 = j0 + 2 + v
                    add_rank1(i2, j2, k0 + 2 + w, 1, c); c += 1
                    
                    # (-A11[u][v]) * B10[v][w]
                    i3 = i0 + 2 + u
                    j3 = j0 + 2 + v
                    add_rank1(i3, j3, k0 + w, -1, c); c += 1
                    
                    # (-A11[u][v]) * B11[v][w]
                    i4 = i0 + 2 + u
                    j4 = j0 + 2 + v
                    add_rank1(i4, j4, k0 + 2 + w, -1, c); c += 1
        
        return c
    
    # Add Strassen for first 4x4x4 block (k=0..3)
    col = add_strassen_4x4(0, 0, 0, col)
    
    # Add Strassen for second 4x4x4 block (k=4..7)
    col = add_strassen_4x4(0, 0, 4, col)
    
    # Add schoolbook for 4x4x1 block (k=8)
    for i in range(n):
        for j in range(m):
            add_rank1(i, j, 8, 1, col)
            col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
