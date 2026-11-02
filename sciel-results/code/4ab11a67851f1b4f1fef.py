"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition: 4 independent 2x2 blocks, each rank 7 -> total rank 28
    n, m, p = DIMENSIONS
    num_blocks = p // m  # 4 blocks of size 2x2
    rank = 7 * num_blocks  # 28
    
    # A is 2x2: indices (i,j) -> row i, col j
    # A indices: a00, a01, a10, a11
    # B is 2x8: for block b, columns are 2b, 2b+1
    #   b00, b01 (row 0, cols 2b, 2b+1)
    #   b10, b11 (row 1, cols 2b, 2b+1)
    # C is 8x2: W[k*n+i, r] where k is column of B, i is row of C
    #   For block b, C rows are 2b, 2b+1
    #   C_{i,k} = sum_j A_{i,j} * B_{j,k}
    #   W index: k*n + i where k in {2b, 2b+1}, i in {0,1}
    #   So for block b: W[2b*2 + 0] = C_{0,2b}, W[2b*2+1] = C_{1,2b}, W[(2b+1)*2+0] = C_{0,2b+1}, W[(2b+1)*2+1] = C_{1,2b+1}
    
    # U shape: (n*m, rank) = (4, 28)
    # V shape: (m*p, rank) = (16, 28)
    # W shape: (p*n, rank) = (16, 28)
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen 7 multiplications for 2x2 * 2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each block, define the 7 terms
    # A coefficients (row, col) for each M:
    # M1: A00(+1), A11(+1)
    # M2: A10(+1), A11(+1)
    # M3: A00(+1)
    # M4: A11(+1)
    # M5: A00(+1), A01(+1)
    # M6: A10(+1), A00(-1)
    # M7: A01(+1), A11(-1)
    
    # A index mapping: (i,j) -> i*m + j = i*2 + j
    # A00 -> 0, A01 -> 1, A10 -> 2, A11 -> 3
    
    # B coefficients for each M (local block indices):
    # M1: B00(+1), B11(+1)
    # M2: B00(+1)
    # M3: B10(+1), B11(-1)
    # M4: B01(+1), B00(-1)
    # M5: B11(+1)
    # M6: B00(+1), B01(+1)
    # M7: B10(+1), B11(+1)
    
    # B index mapping for block b: 
    # B00 -> row 0, col 2b -> index 0*p + 2b = 2b
    # B01 -> row 0, col 2b+1 -> index 2b + 1
    # B10 -> row 1, col 2b -> index 1*p + 2b = 8 + 2b
    # B11 -> row 1, col 2b+1 -> index 8 + 2b + 1 = 9 + 2b
    
    # C coefficients for each M:
    # M1: C00(+1), C11(+1)
    # M2: C10(+1), C11(-1)
    # M3: C01(+1), C11(+1)
    # M4: C00(+1), C10(+1)
    # M5: C00(-1), C01(+1)
    # M6: C11(+1)
    # M7: C00(+1)
    
    # C index mapping: W[k*n + i] where k is B column, i is C row
    # For block b:
    # C00 -> k=2b, i=0 -> 2b*2 + 0 = 4b
    # C01 -> k=2b+1, i=0 -> (2b+1)*2 + 0 = 4b+2
    # C10 -> k=2b, i=1 -> 2b*2 + 1 = 4b+1
    # C11 -> k=2b+1, i=1 -> (2b+1)*2 + 1 = 4b+3
    
    # A coefficients per M: list of (a_idx, coeff)
    A_coeffs = [
        [(0, 1), (3, 1)],       # M1
        [(2, 1), (3, 1)],       # M2
        [(0, 1)],               # M3
        [(3, 1)],               # M4
        [(0, 1), (1, 1)],       # M5
        [(2, 1), (0, -1)],      # M6
        [(1, 1), (3, -1)],      # M7
    ]
    
    # B coefficients per M: list of (local_b_idx, coeff) where local_b_idx: 0=B00,1=B01,2=B10,3=B11
    B_coeffs = [
        [(0, 1), (3, 1)],       # M1
        [(0, 1)],               # M2
        [(2, 1), (3, -1)],      # M3
        [(1, 1), (0, -1)],      # M4
        [(3, 1)],               # M5
        [(0, 1), (1, 1)],       # M6
        [(2, 1), (3, 1)],       # M7
    ]
    
    # C coefficients per M: list of (local_c_idx, coeff) where local_c_idx: 0=C00,1=C01,2=C10,3=C11
    C_coeffs = [
        [(0, 1), (3, 1)],       # M1
        [(2, 1), (3, -1)],      # M2
        [(1, 1), (3, 1)],       # M3
        [(0, 1), (2, 1)],       # M4
        [(0, -1), (1, 1)],      # M5
        [(3, 1)],               # M6
        [(0, 1)],               # M7
    ]
    
    for b in range(num_blocks):
        for t in range(7):
            r = b * 7 + t
            
            # Fill U
            for (a_idx, coeff) in A_coeffs[t]:
                U[a_idx][r] = coeff
            
            # Fill V
            for (local_b_idx, coeff) in B_coeffs[t]:
                # local_b_idx: 0=B00,1=B01,2=B10,3=B11
                # B global index = row * p + col
                # row = local_b_idx // 2, col = 2*b + (local_b_idx % 2)
                b_row = local_b_idx // 2
                b_col = 2 * b + (local_b_idx % 2)
                b_global = b_row * p + b_col
                V[b_global][r] = coeff
            
            # Fill W
            for (local_c_idx, coeff) in C_coeffs[t]:
                # local_c_idx: 0=C00,1=C01,2=C10,3=C11
                # C: i = local_c_idx // 2 (C row), k = 2*b + (local_c_idx % 2) (B col)
                c_i = local_c_idx // 2
                c_k = 2 * b + (local_c_idx % 2)
                w_idx = c_k * n + c_i
                W[w_idx][r] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

