"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 2x5 * 5x6 matrix multiplication using Strassen for 2x2x2 sub-blocks.
    
    Tile:
    - A (2x5): columns split as [0:2], [2:4], [4:5] -> sizes 2, 2, 1
    - B (5x6): rows split as [0:2], [2:4], [4:5] -> sizes 2, 2, 1
               cols split as [0:2], [2:4], [4:6] -> sizes 2, 2, 2
    
    For each (A_col_group, B_row_group, B_col_group) triple:
    - If all sizes are 2: use Strassen (rank 7)
    - Otherwise: use schoolbook
    
    Total schoolbook rank: 2*5*6 = 60
    Number of 2x2x2 blocks: 2*2*2 = 8 (A_col_groups with size 2: 2, B_row_groups with size 2: 2, B_col_groups with size 2: 2)
    Savings: 8 * (8-7) = 8
    Total rank: 60 - 8 = 52
    """
    n, m, p = 2, 5, 6  # A is n x m, B is m x p
    
    # Define tiling
    # A columns: [0:2], [2:4], [4:5]
    a_col_splits = [(0, 2), (2, 4), (4, 5)]
    # B rows: [0:2], [2:4], [4:5]
    b_row_splits = [(0, 2), (2, 4), (4, 5)]
    # B cols: [0:2], [2:4], [4:6]
    b_col_splits = [(0, 2), (2, 4), (4, 6)]
    
    # Count total rank
    total_rank = 0
    blocks = []  # list of (a_col_range, b_row_range, b_col_range, use_strassen)
    for a_start, a_end in a_col_splits:
        a_size = a_end - a_start
        for b_row_start, b_row_end in b_row_splits:
            b_row_size = b_row_end - b_row_start
            for b_col_start, b_col_end in b_col_splits:
                b_col_size = b_col_end - b_col_start
                if a_size == 2 and b_row_size == 2 and b_col_size == 2:
                    blocks.append((a_start, a_end, b_row_start, b_row_end, b_col_start, b_col_end, True))
                    total_rank += 7
                else:
                    blocks.append((a_start, a_end, b_row_start, b_row_end, b_col_start, b_col_end, False))
                    total_rank += a_size * b_row_size * b_col_size
    
    rank = total_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    for (a_start, a_end, b_row_start, b_row_end, b_col_start, b_col_end, use_strassen) in blocks:
        if use_strassen:
            # Strassen for 2x2 * 2x2
            # A block: A[a_start:a_end, a_start:a_end] -> rows i=0,1; cols j=0,1 (local)
            # B block: B[b_row_start:b_row_end, b_col_start:b_col_end] -> rows j=0,1; cols k=0,1 (local)
            # C block: C[i, k] for i=0,1; k=0,1
            # 
            # Strassen M1..M7:
            # M1 = (a00 + a11) * (b00 + b11)
            # M2 = (a10 + a11) * b00
            # M3 = a00 * (b01 - b11)
            # M4 = a11 * (b10 - b00)
            # M5 = (a00 + a01) * b11
            # M6 = (a10 - a00) * (b00 + b01)
            # M7 = (a01 - a11) * (b10 + b11)
            #
            # C00 = M1 - M4 + M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            #
            # Each M_t is a product of a linear combination of A entries and a linear combination of B entries.
            # In the tensor decomposition, each M_t contributes one rank-1 term.
            #
            # For M_t = (sum of alpha * A[i,j]) * (sum of beta * B[j,k]):
            # U[i*m+j, col] += alpha for each (i,j) in A
            # V[j*p+k, col] += beta for each (j,k) in B  
            # W[k*n+i, col] += 1 for each (i,k) in C (the output coefficient)
            
            # Define Strassen terms: (list of (A_coeffs, B_coeffs, C_coeffs))
            # A_coeffs: dict (i,j) -> coeff, B_coeffs: dict (j,k) -> coeff, C_coeffs: dict (i,k) -> coeff
            
            # M1 = (A00 + A11) * (B00 + B11)
            # C00 += M1, C11 += M1
            strassen_terms = [
                # (A_coeffs, B_coeffs, C_coeffs)
                # M1
                ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
                # M2 = (A10 + A11) * B00
                # C10 += M2, C11 -= M2
                ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
                # M3 = A00 * (B01 - B11)
                # C01 += M3, C11 += M3
                ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
                # M4 = A11 * (B10 - B00)
                # C00 -= M4, C10 += M4
                ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): -1, (1,0): 1}),
                # M5 = (A00 + A01) * B11
                # C00 += M5, C01 += M5
                ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): 1, (0,1): 1}),
                # M6 = (A10 - A00) * (B00 + B01)
                # C11 += M6
                ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
                # M7 = (A01 - A11) * (B10 + B11)
                # C00 += M7
                ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
            ]
            
            for (a_coeffs, b_coeffs, c_coeffs) in strassen_terms:
                col = col_offset
                for (li, lj), a_val in a_coeffs.items():
                    global_i = a_start + li
                    global_j = a_start + lj  # A cols map to B rows
                    U[global_i * m + global_j][col] = a_val
                for (lj, lk), b_val in b_coeffs.items():
                    global_j = b_row_start + lj
                    global_k = b_col_start + lk
                    V[global_j * p + global_k][col] = b_val
                for (li, lk), c_val in c_coeffs.items():
                    global_i = a_start + li
                    global_k = b_col_start + lk
                    W[global_k * n + global_i][col] = c_val
                col_offset += 1
        else:
            # Schoolbook for this block
            a_size = a_end - a_start
            b_row_size = b_row_end - b_row_start
            b_col_size = b_col_end - b_col_start
            for li in range(2):  # i index (A rows are always 0,1)
                for lj in range(b_row_size):  # j index
                    for lk in range(b_col_size):  # k index
                        col = col_offset
                        global_i = li  # A row
                        global_j = a_start + lj  # A col = B row
                        global_k = b_col_start + lk  # B col
                        U[global_i * m + global_j][col] = 1
                        V[global_j * p + global_k][col] = 1
                        W[global_k * n + global_i][col] = 1
                        col_offset += 1
    
    assert col_offset == rank, f"col_offset {col_offset} != rank {rank}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
