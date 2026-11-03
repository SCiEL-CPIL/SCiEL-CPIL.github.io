"""Matrix multiplication 3x5x6 - block-recursive decomposition with Strassen."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-84 tensor decomposition for 3x5x6 matrix multiplication.
    
    Split: n=3 as 2+1, m=5 as 2+2+1, p=6 as 2+2+2
    Blocks and their ranks:
      (2,2,2): rank 7 (Strassen) - 6 blocks
      (2,1,2): rank 4 (schoolbook) - 3 blocks  
      (1,2,2): rank 4 (schoolbook) - 6 blocks
      (1,1,2): rank 2 (schoolbook) - 3 blocks
    Total: 42 + 12 + 24 + 6 = 84
    """
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Define splits
    n_splits = [2, 1]       # n = 3
    m_splits = [2, 2, 1]   # m = 5
    p_splits = [2, 2, 2]   # p = 6
    
    # Compute starting offsets
    n_offsets = []
    off = 0
    for s in n_splits:
        n_offsets.append(off)
        off += s
    m_offsets = []
    off = 0
    for s in m_splits:
        m_offsets.append(off)
        off += s
    p_offsets = []
    off = 0
    for s in p_splits:
        p_offsets.append(off)
        off += s
    
    # Collect all block decompositions
    blocks = []
    for ni in range(len(n_splits)):
        for mi in range(len(m_splits)):
            for pi in range(len(p_splits)):
                ni_size = n_splits[ni]
                mi_size = m_splits[mi]
                pi_size = p_splits[pi]
                blocks.append((ni, mi, pi, ni_size, mi_size, pi_size,
                               n_offsets[ni], m_offsets[mi], p_offsets[pi]))
    
    # Generate U, V, W for each block type
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
        # Strassen's 7 products:
        # M1 = (A22 - A11)(B11 + B12)
        # M2 = (A11 + A22)(B22 - B12)
        # M3 = (A11 + A12)B22
        # M4 = A22(B11 - B22)
        # M5 = (A11 + A22)(B11 + B22)
        # M6 = (A11 - A21)(B11 + B21)
        # M7 = (A12 - A22)(B12 + B22)
        #
        # C11 = M5 + M4 - M2 + M6
        # C12 = M1 + M2
        # C21 = M3 + M5
        # C22 = M1 - M4 + M5 + M7
        #
        # U: 4 rows (A11,A12,A21,A22) x 7 cols
        # V: 4 rows (B11,B12,B21,B22) x 7 cols
        # W: 4 rows (C11,C12,C21,C22) x 7 cols
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A22 - A11)(B11 + B12)
        U[3][0] = 1   # A22
        U[0][0] = -1  # A11
        V[0][0] = 1   # B11
        V[1][0] = 1   # B12
        W[1][0] = 1   # C12
        
        # M2 = (A11 + A22)(B22 - B12)
        U[0][1] = 1   # A11
        U[3][1] = 1   # A22
        V[3][1] = 1   # B22
        V[1][1] = -1  # B12
        W[0][1] = -1  # C11
        W[1][1] = 1   # C12
        
        # M3 = (A11 + A12)B22
        U[0][2] = 1   # A11
        U[1][2] = 1   # A12
        V[3][2] = 1   # B22
        W[2][2] = 1   # C21
        
        # M4 = A22(B11 - B22)
        U[3][3] = 1   # A22
        V[0][3] = 1   # B11
        V[3][3] = -1  # B22
        W[0][3] = 1   # C11
        W[3][3] = -1  # C22
        
        # M5 = (A11 + A22)(B11 + B22)
        U[0][4] = 1   # A11
        U[3][4] = 1   # A22
        V[0][4] = 1   # B11
        V[3][4] = 1   # B22
        W[0][4] = 1   # C11
        W[2][4] = 1   # C21
        W[3][4] = 1   # C22
        
        # M6 = (A11 - A21)(B11 + B21)
        U[0][5] = 1   # A11
        U[2][5] = -1  # A21
        V[0][5] = 1   # B11
        V[2][5] = 1   # B21
        W[0][5] = 1   # C11
        
        # M7 = (A12 - A22)(B12 + B22)
        U[1][6] = 1   # A12
        U[3][6] = -1  # A22
        V[1][6] = 1   # B12
        V[3][6] = 1   # B22
        W[3][6] = 1   # C22
        
        return U, V, W
    
    def schoolbook_2x1x2():
        """Return U, V, W for 2x1x2 schoolbook (rank 4)."""
        # A is 2x1, B is 1x2, C is 2x2
        # C[0,0] = A[0,0]*B[0,0]
        # C[0,1] = A[0,0]*B[0,1]
        # C[1,0] = A[1,0]*B[0,0]
        # C[1,1] = A[1,0]*B[0,1]
        U = [[0]*4 for _ in range(2)]
        V = [[0]*4 for _ in range(2)]
        W = [[0]*4 for _ in range(4)]
        
        # Term 0: A[0,0]*B[0,0] -> C[0,0]
        U[0][0] = 1
        V[0][0] = 1
        W[0][0] = 1
        
        # Term 1: A[0,0]*B[0,1] -> C[0,1]
        U[0][1] = 1
        V[1][1] = 1
        W[1][1] = 1
        
        # Term 2: A[1,0]*B[0,0] -> C[1,0]
        U[1][2] = 1
        V[0][2] = 1
        W[2][2] = 1
        
        # Term 3: A[1,0]*B[0,1] -> C[1,1]
        U[1][3] = 1
        V[1][3] = 1
        W[3][3] = 1
        
        return U, V, W
    
    def schoolbook_1x2x2():
        """Return U, V, W for 1x2x2 schoolbook (rank 4)."""
        # A is 1x2, B is 2x2, C is 1x2
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        U = [[0]*4 for _ in range(2)]
        V = [[0]*4 for _ in range(4)]
        W = [[0]*4 for _ in range(2)]
        
        # Term 0: A[0,0]*B[0,0] -> C[0,0]
        U[0][0] = 1
        V[0][0] = 1
        W[0][0] = 1
        
        # Term 1: A[0,1]*B[1,0] -> C[0,0]
        U[1][1] = 1
        V[2][1] = 1
        W[0][1] = 1
        
        # Term 2: A[0,0]*B[0,1] -> C[0,1]
        U[0][2] = 1
        V[1][2] = 1
        W[1][2] = 1
        
        # Term 3: A[0,1]*B[1,1] -> C[0,1]
        U[1][3] = 1
        V[3][3] = 1
        W[1][3] = 1
        
        return U, V, W
    
    def schoolbook_1x1x2():
        """Return U, V, W for 1x1x2 schoolbook (rank 2)."""
        # A is 1x1, B is 1x2, C is 1x2
        U = [[0]*2 for _ in range(1)]
        V = [[0]*2 for _ in range(2)]
        W = [[0]*2 for _ in range(2)]
        
        # Term 0: A[0,0]*B[0,0] -> C[0,0]
        U[0][0] = 1
        V[0][0] = 1
        W[0][0] = 1
        
        # Term 1: A[0,0]*B[0,1] -> C[0,1]
        U[0][1] = 1
        V[1][1] = 1
        W[1][1] = 1
        
        return U, V, W
    
    # Initialize full U, V, W
    total_rank = sum(
        (7 if (bs[3]==2 and bs[4]==2 and bs[5]==2) else
         4 if (bs[3]==2 and bs[4]==1 and bs[5]==2) else
         4 if (bs[3]==1 and bs[4]==2 and bs[5]==2) else
         2)
        for bs in blocks
    )
    
    U_full = [[0]*total_rank for _ in range(n*m)]  # 15 x total_rank
    V_full = [[0]*total_rank for _ in range(m*p)]  # 30 x total_rank
    W_full = [[0]*total_rank for _ in range(p*n)]  # 18 x total_rank
    
    col_offset = 0
    for (ni, mi, pi, ni_size, mi_size, pi_size, n_off, m_off, p_off) in blocks:
        if ni_size == 2 and mi_size == 2 and pi_size == 2:
            rank = 7
            Ub, Vb, Wb = strassen_2x2x2()
        elif ni_size == 2 and mi_size == 1 and pi_size == 2:
            rank = 4
            Ub, Vb, Wb = schoolbook_2x1x2()
        elif ni_size == 1 and mi_size == 2 and pi_size == 2:
            rank = 4
            Ub, Vb, Wb = schoolbook_1x2x2()
        elif ni_size == 1 and mi_size == 1 and pi_size == 2:
            rank = 2
            Ub, Vb, Wb = schoolbook_1x1x2()
        else:
            raise ValueError(f"Unknown block type: {ni_size}x{mi_size}x{pi_size}")
        
        # Embed into full matrices
        for r in range(ni_size * mi_size):
            i = r // mi_size
            j = r % mi_size
            full_row = (n_off + i) * m + (m_off + j)
            for c in range(rank):
                U_full[full_row][col_offset + c] = Ub[r][c]
        
        for r in range(mi_size * pi_size):
            j = r // pi_size
            k = r % pi_size
            full_row = (m_off + j) * p + (p_off + k)
            for c in range(rank):
                V_full[full_row][col_offset + c] = Vb[r][c]
        
        for r in range(pi_size * ni_size):
            k = r // ni_size
            i = r % ni_size
            full_row = (p_off + k) * n + (n_off + i)
            for c in range(rank):
                W_full[full_row][col_offset + c] = Wb[r][c]
        
        col_offset += rank
    
    return {"U": U_full, "V": V_full, "W": W_full}
# EVOLVE-BLOCK-END
