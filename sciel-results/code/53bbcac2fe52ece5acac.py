"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-21 decomposition using Strassen's algorithm for each 2x2x2 block.
    The 2x2x6 multiplication is viewed as 3 independent 2x2x2 multiplications
    sharing the same A matrix, so the A-side Strassen factors are reused.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=6
    
    # Strassen's 7 products for 2x2x2:
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C = A*B
    #
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6 + M7
    
    # A-side factors (7 combinations of A entries):
    # Index: (A11, A12, A21, A22)
    a_factors = [
        (1, 0, 0, 1),   # M1: A11+A22
        (0, 0, 1, 1),   # M2: A21+A22
        (1, 0, 0, 0),   # M3: A11
        (0, 0, 0, 1),   # M4: A22
        (1, 1, 0, 0),   # M5: A11+A12
        (-1, 0, 1, 0),  # M6: A21-A11
        (0, 1, 0, -1),  # M7: A12-A22
    ]
    
    # B-side factors for each product (B11, B12, B21, B22):
    # For a given 2x2 block of B
    b_factors = [
        (1, 0, 0, 1),   # M1: B11+B22
        (1, 0, 0, 0),   # M2: B11
        (0, 0, 1, -1),  # M3: B21-B22
        (0, 1, 0, -1),  # M4: B12-B11
        (0, 0, 0, 1),   # M5: B22
        (1, 1, 0, 0),   # M6: B11+B12
        (0, 0, 1, 1),   # M7: B21+B22
    ]
    
    # C-side reconstruction: for each output C entry, which M's contribute with what sign
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6 + M7
    #
    # c_factors[output_idx][m_idx] = coefficient
    # Output indices: C11=0, C12=1, C21=2, C22=3
    c_factors = [
        [1, 0, 0, -1, 1, 0, 1],   # C11
        [0, 0, 1, 0, 1, 0, 0],    # C12
        [0, 1, 0, 1, 0, 0, 0],    # C21
        [1, 0, -1, 0, 0, 1, 1],   # C22
    ]
    
    num_blocks = p // m  # 3 blocks of 2 columns each
    rank = num_blocks * 7  # 21
    
    # Build U: (n*m, rank) = (4, 21)
    U = [[0] * rank for _ in range(n * m)]
    # Build V: (m*p, rank) = (12, 21)
    V = [[0] * rank for _ in range(m * p)]
    # Build W: (p*n, rank) = (12, 21)
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for block in range(num_blocks):
        for m_idx in range(7):
            # U: A-side factor
            for a_idx in range(4):
                U[a_idx][col] = a_factors[m_idx][a_idx]
            
            # V: B-side factor for this block
            # B entries for block: rows 0,1 of B, columns 2*block, 2*block+1
            # V row index: j*p + k where j is row of B (0 or 1), k is column of B
            # B11 = B[0, 2*block], B12 = B[0, 2*block+1]
            # B21 = B[1, 2*block], B22 = B[1, 2*block+1]
            b_entry = b_factors[m_idx]
            # b_entry = (coeff_B11, coeff_B12, coeff_B21, coeff_B22)
            V[0 * p + (2 * block + 0)][col] = b_entry[0]   # B11
            V[0 * p + (2 * block + 1)][col] = b_entry[1]   # B12
            V[1 * p + (2 * block + 0)][col] = b_entry[2]   # B21
            V[1 * p + (2 * block + 1)][col] = b_entry[3]   # B22
            
            # W: C-side reconstruction
            # W row index: k*n + i where k is column of C, i is row of C
            # C entries for block: C[0, 2*block], C[0, 2*block+1], C[1, 2*block], C[1, 2*block+1]
            # C11 = C[0, 2*block], C12 = C[0, 2*block+1]
            # C21 = C[1, 2*block], C22 = C[1, 2*block+1]
            for c_idx in range(4):
                coeff = c_factors[c_idx][m_idx]
                if coeff != 0:
                    # c_idx: 0=C11, 1=C12, 2=C21, 3=C22
                    if c_idx == 0:
                        k, i = 2 * block + 0, 0
                    elif c_idx == 1:
                        k, i = 2 * block + 1, 0
                    elif c_idx == 2:
                        k, i = 2 * block + 0, 1
                    else:
                        k, i = 2 * block + 1, 1
                    W[k * n + i][col] = coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
