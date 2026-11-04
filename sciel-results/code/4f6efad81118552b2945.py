"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based block decomposition for 4x5 x 5x6 matrix multiplication.
    # Partition: A (4x5) as 2 blocks of 2x? in rows, 3 blocks in cols (2,2,1)
    #             B (5x6) as 3 blocks in rows (2,2,1), 3 blocks in cols (2,2,2)
    # For each (ib, jb, kb) we multiply a (2 x b_j) submatrix by a (b_j x 2) submatrix
    #   where b_j = 2 for jb in {0,1}, b_j = 1 for jb = 2
    # If b_j = 2: use Strassen (rank 7) for 2x2 * 2x2
    # If b_j = 1: use schoolbook (rank 4) for 2x1 * 1x2
    # Total rank = 6 * 2 * 7 + 2 * 3 * 4 = 84 + 24 = 108
    import time
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    rank = 108
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    deadline = start_time + 100  # 100 second safety margin
    
    # Column offsets in the 5-col / 6-col partitions
    col_offsets_B = [0, 2, 4]  # B's column offsets (size 2, 2, 1)
    row_offsets_B = [0, 2, 4]  # B's row offsets (size 2, 2, 1)
    col_offsets_A = [0, 2, 4]  # A's column offsets (size 2, 2, 1)
    row_offsets_A = [0, 2]     # A's row offsets (size 2, 2)
    
    def add_strassen(i0, j0, k0):
        """Add Strassen decomposition for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]"""
        nonlocal col
        # A sub-block: rows i0, i0+1; cols j0, j0+1
        # B sub-block: rows j0, j0+1; cols k0, k0+1
        # C sub-block: rows i0, i0+1; cols k0, k0+1
        
        # A indices in U: (i, j) -> U[i*m + j]
        a00 = (i0, j0)
        a01 = (i0, j0+1)
        a10 = (i0+1, j0)
        a11 = (i0+1, j0+1)
        
        # B indices in V: (j, k) -> V[j*p + k]
        b00 = (j0, k0)
        b01 = (j0, k0+1)
        b10 = (j0+1, k0)
        b11 = (j0+1, k0+1)
        
        # C indices in W: (k, i) -> W[k*n + i]  (trace convention W[k*n+i, r])
        c00 = (k0, i0)
        c01 = (k0+1, i0)
        c10 = (k0, i0+1)
        c11 = (k0+1, i0+1)
        
        # m1 = (a00+a11)(b00+b11)
        U[a00[0]*m + a00[1]][col] += 1
        U[a11[0]*m + a11[1]][col] += 1
        V[b00[0]*p + b00[1]][col] += 1
        V[b11[0]*p + b11[1]][col] += 1
        W[c00[1]*n + c00[0]][col] += 1
        W[c01[1]*n + c01[0]][col] += 1
        W[c10[1]*n + c10[0]][col] += 1
        W[c11[1]*n + c11[0]][col] += 1
        col += 1
        
        # m2 = (a10+a11)b00
        U[a10[0]*m + a10[1]][col] += 1
        U[a11[0]*m + a11[1]][col] += 1
        V[b00[0]*p + b00[1]][col] += 1
        W[c10[1]*n + c10[0]][col] += 1
        W[c11[1]*n + c11[0]][col] += 1
        col += 1
        
        # m3 = a00(b01-b11)
        U[a00[0]*m + a00[1]][col] += 1
        V[b01[0]*p + b01[1]][col] += 1
        V[b11[0]*p + b11[1]][col] -= 1
        W[c00[1]*n + c00[0]][col] += 1
        W[c01[1]*n + c01[0]][col] += 1
        col += 1
        
        # m4 = a11(b10-b00)
        U[a11[0]*m + a11[1]][col] += 1
        V[b10[0]*p + b10[1]][col] += 1
        V[b00[0]*p + b00[1]][col] -= 1
        W[c10[1]*n + c10[0]][col] += 1
        W[c11[1]*n + c11[0]][col] += 1
        col += 1
        
        # m5 = (a00+a01)b11
        U[a00[0]*m + a00[1]][col] += 1
        U[a01[0]*m + a01[1]][col] += 1
        V[b11[0]*p + b11[1]][col] += 1
        W[c00[1]*n + c00[0]][col] += 1
        W[c01[1]*n + c01[0]][col] += 1
        col += 1
        
        # m6 = (a10-a00)(b00+b01)
        U[a10[0]*m + a10[1]][col] += 1
        U[a00[0]*m + a00[1]][col] -= 1
        V[b00[0]*p + b00[1]][col] += 1
        V[b01[0]*p + b01[1]][col] += 1
        W[c11[1]*n + c11[0]][col] += 1
        col += 1
        
        # m7 = (a01-a11)(b10+b11)
        U[a01[0]*m + a01[1]][col] += 1
        U[a11[0]*m + a11[1]][col] -= 1
        V[b10[0]*p + b10[1]][col] += 1
        V[b11[0]*p + b11[1]][col] += 1
        W[c00[1]*n + c00[0]][col] += 1
        col += 1
    
    def add_schoolbook_2x1x2(i0, j0, k0):
        """Add schoolbook for A[i0:i0+2, j0:j0+1] * B[j0:j0+1, k0:k0+2] -> C[i0:i0+2, k0:k0+2]"""
        nonlocal col
        for i_off in range(2):
            for k_off in range(2):
                i = i0 + i_off
                k = k0 + k_off
                U[i*m + j0][col] += 1
                V[j0*p + k][col] += 1
                W[k*n + i][col] += 1
                col += 1
    
    # Iterate over all blocks
    # ib in {0,1} (2 row blocks of size 2 for A)
    # jb in {0,1,2} (3 column blocks for A: sizes 2,2,1 = 3 column blocks for B rows)
    # kb in {0,1,2} (3 column blocks for B: sizes 2,2,2)
    for ib in range(2):
        for jb in range(3):
            for kb in range(3):
                i0 = ib * 2
                j0 = col_offsets_A[jb]
                k0 = col_offsets_B[kb]
                
                if jb < 2:
                    # A block is 2x2, B block is 2x2, use Strassen (rank 7)
                    add_strassen(i0, j0, k0)
                else:
                    # A block is 2x1, B block is 1x2, use schoolbook (rank 4)
                    add_schoolbook_2x1x2(i0, j0, k0)
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
