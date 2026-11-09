"""Combined program: schoolbook fallback + Strassen-based block decomposition for 3x4x7 matrix multiplication."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    DEADLINE = start_time + 115  # 120s limit, keep 5s margin

    n, m, p = DIMENSIONS  # 3, 4, 7

    def schoolbook():
        """Standard schoolbook decomposition of rank n*m*p."""
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

    best = schoolbook()

    def verify(U, V, W, n, m, p):
        """Verify exactness: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    # Precompute Strassen 2x2x2 decomposition (computed once)
    _strassen_222_cache = None
    def strassen_222():
        """
        Strassen decomposition for 2x2x2 matrix multiplication.
        Returns (U, V, W) where:
          U: shape (4, 7) - coefficients for A entries
          V: shape (4, 7) - coefficients for B entries
          W: shape (4, 7) - coefficients for C entries
        Indexing: U[i*2+j, c] for A_{i,j}, V[j*2+k, c] for B_{j,k}, W[k*2+i, c] for C_{i,k}
        """
        nonlocal _strassen_222_cache
        if _strassen_222_cache is not None:
            return _strassen_222_cache
        
        rank = 7
        U = [[0] * rank for _ in range(4)]
        V = [[0] * rank for _ in range(4)]
        W = [[0] * rank for _ in range(4)]

        # W indexing: W[k*n + i, c] where C[i,k], n=2
        # C[0,0] -> W[0], C[0,1] -> W[2], C[1,0] -> W[1], C[1,1] -> W[3]

        # M1: (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1   # A00, A11
        V[0][0] = 1; V[3][0] = 1   # B00, B11
        W[0][0] = 1                # C00 += M1

        # M2: (A10+A11)(B00)
        U[2][1] = 1; U[3][1] = 1   # A10, A11
        V[0][1] = 1                # B00
        W[1][1] = 1                # C10 += M2

        # M3: A00(B01-B11)
        U[0][2] = 1                # A00
        V[1][2] = 1; V[3][2] = -1  # B01 - B11
        W[2][2] = 1                # C01 += M3

        # M4: A11(B10-B00)
        U[3][3] = 1                # A11
        V[2][3] = 1; V[0][3] = -1  # B10 - B00
        W[0][3] = 1                # C00 += M4
        W[1][3] = 1                # C10 += M4

        # M5: (A00+A01)(B11)
        U[0][4] = 1; U[1][4] = 1   # A00, A01
        V[3][4] = 1                # B11
        W[2][4] = 1                # C01 += M5
        W[0][4] = -1               # C00 -= M5

        # M6: (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1  # A10 - A00
        V[0][5] = 1; V[1][5] = 1   # B00 + B01
        W[3][5] = 1                # C11 += M6

        # M7: (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1  # A01 - A11
        V[2][6] = 1; V[3][6] = 1   # B10 + B11
        W[0][6] = 1                # C00 += M7
        W[3][6] = 1                # C11 += M7

        _strassen_222_cache = (U, V, W)
        return U, V, W

    def block_decompose():
        """
        Block decomposition for 3x4x7 using Strassen on 2x2x2 sub-blocks.
        
        Strategy:
        - Split n=3 into (2,1), m=4 into (2,2), p=7 into (2,2,2,1)
        - For each combination, use appropriate decomposition
        - 2x2x2 blocks use Strassen (rank 7)
        - Other blocks use schoolbook
        
        Total rank: 78
        """
        # This is getting complex. Let me use a simpler approach:
        # Just use Strassen on the 2x2 parts where possible.
        
        # Actually, let's try a different decomposition strategy.
        # Use the fact that we can embed smaller matrix multiplications.
        
        # For now, let's try to construct a decomposition with rank < 84
        # by using Strassen-like ideas on sub-blocks.
        
        # Split A (3x4) into A_top (2x4) and A_bot (1x4)
        # Split B (4x7) into B_left (4x2), B_mid (4x2), B_mid2 (4x2), B_right (4x1)
        # C = A*B = [A_top; A_bot] * [B_left, B_mid, B_mid2, B_right]
        
        # C_top_left (2x2) = A_top * B_left
        # C_top_mid (2x2) = A_top * B_mid
        # C_top_mid2 (2x2) = A_top * B_mid2
        # C_top_right (2x1) = A_top * B_right
        # C_bot_left (1x2) = A_bot * B_left
        # C_bot_mid (1x2) = A_bot * B_mid
        # C_bot_mid2 (1x2) = A_bot * B_mid2
        # C_bot_right (1x1) = A_bot * B_right
        
        # For 2x2x2 blocks (C_top_left, C_top_mid, C_top_mid2): use Strassen (rank 7 each)
        # But wait, these are 2x4 * 4x2, not 2x2*2x2.
        # I need to split the middle dimension too.
        
        # Let me reconsider. The problem is 3x4x7, so A is 3x4, B is 4x7.
        # To use Strassen, I need 2x2x2 sub-problems, which means I need to split
        # the middle dimension (4) into 2+2, and then each 2x2x2 uses Strassen.
        
        # Split m=4 into m1=2, m2=2
        # A = [A1, A2] where A1, A2 are 3x2
        # B = [B1; B2] where B1, B2 are 2x7
        # C = A1*B1 + A2*B2
        
        # Now A1*B1 is 3x2 * 2x7 = 3x7
        # A2*B2 is 3x2 * 2x7 = 3x7
        
        # For 3x2*2x7, split n=3 into 2+1:
        # A1 = [A1_top; A1_bot] where A1_top is 2x2, A1_bot is 1x2
        # C1 = A1_top * B1 + A1_bot * B1
        # A1_top * B1 is 2x2 * 2x7
        # A1_bot * B1 is 1x2 * 2x7
        
        # For 2x2*2x7, split p=7 into 2+2+2+1:
        # B1 = [B1_0, B1_1, B1_2, B1_3] where B1_0,B1_1,B1_2 are 2x2, B1_3 is 2x1
        # A1_top * B1 = [A1_top*B1_0, A1_top*B1_1, A1_top*B1_2, A1_top*B1_3]
        # Each A1_top*B1_i for i=0,1,2 is 2x2*2x2 -> Strassen (rank 7)
        # A1_top*B1_3 is 2x2*2x1 -> schoolbook (rank 4)
        
        # For 1x2*2x7, split p=7 into 2+2+2+1:
        # A1_bot * B1 = [A1_bot*B1_0, A1_bot*B1_1, A1_bot*B1_2, A1_bot*B1_3]
        # Each A1_bot*B1_i for i=0,1,2 is 1x2*2x2 -> schoolbook (rank 4)
        # A1_bot*B1_3 is 1x2*2x1 -> schoolbook (rank 2)
        
        # So for A1*B1:
        # 3 * 7 (Strassen) + 4 (schoolbook) + 3 * 4 (schoolbook) + 2 (schoolbook)
        # = 21 + 4 + 12 + 2 = 39
        
        # Similarly for A2*B2: 39
        
        # Total: 78
        
        # Now let's construct this.
        
        n, m, p = 3, 4, 7
        # Split m into 2+2
        m1, m2 = 2, 2
        # Split n into 2+1
        n1, n2 = 2, 1
        # Split p into 2+2+2+1
        p_splits = [2, 2, 2, 1]
        
        # We'll build U, V, W incrementally
        # U: (n*m, rank) = (12, rank)
        # V: (m*p, rank) = (28, rank)
        # W: (p*n, rank) = (21, rank)
        
        # First, let's figure out the total rank
        # For each of the 2 m-blocks:
        #   For the 2x2 part of n:
        #     For each of 3 p-blocks of size 2: Strassen (rank 7)
        #     For the 1 p-block of size 1: schoolbook (rank 4)
        #   For the 1x1 part of n:
        #     For each of 3 p-blocks of size 2: schoolbook (rank 4)
        #     For the 1 p-block of size 1: schoolbook (rank 2)
        # Per m-block: 3*7 + 4 + 3*4 + 2 = 21+4+12+2 = 39
        # Total: 2 * 39 = 78
        
        total_rank = 78
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        
        # Helper to add a 2x2x2 Strassen block
        def add_strassen_block(U, V, W, i_offset, j_offset, k_offset, col_start):
            """Add Strassen decomposition for a 2x2x2 block at given offsets."""
            sU, sV, sW = strassen_222()
            for i in range(2):
                for j in range(2):
                    for c in range(7):
                        U[(i_offset + i) * m + (j_offset + j)][col_start + c] += sU[i * 2 + j][c]
            for j in range(2):
                for k in range(2):
                    for c in range(7):
                        V[(j_offset + j) * p + (k_offset + k)][col_start + c] += sV[j * 2 + k][c]
            for k in range(2):
                for i in range(2):
                    for c in range(7):
                        W[(k_offset + k) * n + (i_offset + i)][col_start + c] += sW[k * 2 + i][c]
            return col_start + 7
        
        # Helper to add a schoolbook block for n_sub x m_sub x p_sub
        def add_schoolbook_block(U, V, W, i_offset, j_offset, k_offset, col_start, n_sub, m_sub, p_sub):
            """Add schoolbook decomposition for an n_sub x m_sub x p_sub block."""
            rank = n_sub * m_sub * p_sub
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        c = col_start + (i * m_sub + j) * p_sub + k
                        U[(i_offset + i) * m + (j_offset + j)][c] = 1
                        V[(j_offset + j) * p + (k_offset + k)][c] = 1
                        W[(k_offset + k) * n + (i_offset + i)][c] = 1
            return col_start + rank
        
        # Process each m-block
        for m_block in range(2):
            j_offset = m_block * m1  # 0 or 2
            
            # Process the 2x2 part of n (i_offset = 0)
            i_offset = 0
            n_sub = 2
            m_sub = m1  # 2
            
            # Process each p-block
            k_offset = 0
            for p_idx, p_sub in enumerate(p_splits):
                if p_sub == 2:
                    # Use Strassen for 2x2x2
                    col = add_strassen_block(U, V, W, i_offset, j_offset, k_offset, col)
                else:
                    # Use schoolbook for 2x2xp_sub
                    col = add_schoolbook_block(U, V, W, i_offset, j_offset, k_offset, col, n_sub, m_sub, p_sub)
                k_offset += p_sub
            
            # Process the 1x1 part of n (i_offset = 2)
            i_offset = 2
            n_sub = 1
            m_sub = m1  # 2
            
            # Process each p-block
            k_offset = 0
            for p_idx, p_sub in enumerate(p_splits):
                # Use schoolbook for 1x2xp_sub
                col = add_schoolbook_block(U, V, W, i_offset, j_offset, k_offset, col, n_sub, m_sub, p_sub)
                k_offset += p_sub
        
        # Verify the result
        if verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        
        # If verification fails, fall back to schoolbook
        return best

    def try_block_decompose(n_splits, m_splits, p_splits):
        """Try a specific block decomposition and return the result if valid."""
        n, m, p = 3, 4, 7
        total_rank = 0
        # Calculate expected rank
        for i_sub in n_splits:
            for j_sub in m_splits:
                for k_sub in p_splits:
                    if i_sub == 2 and j_sub == 2 and k_sub == 2:
                        total_rank += 7  # Strassen
                    else:
                        total_rank += i_sub * j_sub * k_sub  # Schoolbook
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        i_offsets = [0]
        for ns in n_splits[:-1]:
            i_offsets.append(i_offsets[-1] + ns)
        j_offsets = [0]
        for ms in m_splits[:-1]:
            j_offsets.append(j_offsets[-1] + ms)
        k_offsets = [0]
        for ps in p_splits[:-1]:
            k_offsets.append(k_offsets[-1] + ps)
        
        for i_idx, i_sub in enumerate(n_splits):
            for j_idx, j_sub in enumerate(m_splits):
                for k_idx, k_sub in enumerate(p_splits):
                    i_off = i_offsets[i_idx]
                    j_off = j_offsets[j_idx]
                    k_off = k_offsets[k_idx]
                    if i_sub == 2 and j_sub == 2 and k_sub == 2:
                        col = add_strassen_block(U, V, W, i_off, j_off, k_off, col)
                    else:
                        col = add_schoolbook_block(U, V, W, i_off, j_off, k_off, col, i_sub, j_sub, k_sub)
        
        if col != total_rank:
            return None
        
        if verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        return None
    
    # Try the original block decomposition
    try:
        result = block_decompose()
        if len(result["U"][0]) < len(best["U"][0]):
            best = result
    except Exception:
        pass
    
    # Try alternative p-splits with same n=(2,1), m=(2,2)
    alt_p_splits = [
        [2, 2, 2, 1],  # original
        [2, 2, 1, 2],
        [2, 1, 2, 2],
        [1, 2, 2, 2],
    ]
    
    for ps in alt_p_splits:
        if time.time() > DEADLINE - 5:
            break
        try:
            result = try_block_decompose([2, 1], [2, 2], ps)
            if result is not None and len(result["U"][0]) < len(best["U"][0]):
                best = result
        except Exception:
            pass
    
    return best
# EVOLVE-BLOCK-END
