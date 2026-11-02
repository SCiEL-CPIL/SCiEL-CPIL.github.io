"""Matrix multiplication 2x2 by 2x8 - rank 28 via 4x Strassen rank-7 blocks."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=8
    num_blocks = p // m  # 4 blocks of 2x2
    rank_per_block = 7   # Strassen rank for 2x2x2
    total_rank = num_blocks * rank_per_block  # 28
    
    U = [[0] * total_rank for _ in range(n * m)]  # (4, 28)
    V = [[0] * total_rank for _ in range(m * p)]  # (16, 28)
    W = [[0] * total_rank for _ in range(p * n)]  # (16, 28)
    
    # Strassen coefficients for each of the 7 products
    # For A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each Strassen product, define (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: dict from (i,j) to coefficient in A
    # b_coeffs: dict from (j,k_local) to coefficient in B_block
    # c_coeffs: dict from (k_local,i) to coefficient in C_block
    
    strassen_products = [
        # M1: (a00+a11)*(b00+b11)
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ),
        # M2: (a10+a11)*b00
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(0,0): 1}  # contributes to C10
        ),
        # M3: a00*(b01-b11)
        (
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1}  # C01 += M3, C11 += M3
        ),
        # M4: a11*(b10-b00)
        (
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (1,0): 1}  # C00 += M4, C10 += M4
        ),
        # M5: (a00+a01)*b11
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (0,1): 1}  # C00 -= M5, C01 += M5
        ),
        # M6: (a10-a00)*(b00+b01)
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}  # C11 += M6
        ),
        # M7: (a01-a11)*(b10+b11)
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}  # C00 += M7
        ),
    ]
    
    for b in range(num_blocks):
        base_col = b * rank_per_block
        for s in range(rank_per_block):
            col = base_col + s
            a_coeffs, b_coeffs, c_coeffs = strassen_products[s]
            
            # Fill U: U[i*m + j, col] = a_coeff
            for (i, j), coeff in a_coeffs.items():
                U[i * m + j][col] = coeff
            
            # Fill V: V[j*p + k, col] = b_coeff
            # k_local is 0 or 1, actual k = b*m + k_local
            for (j, k_local), coeff in b_coeffs.items():
                k = b * m + k_local
                V[j * p + k][col] = coeff
            
            # Fill W: W[k*n + i, col] = c_coeff
            # k_local is 0 or 1, actual k = b*m + k_local
            for (k_local, i), coeff in c_coeffs.items():
                k = b * m + k_local
                W[k * n + i][col] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
