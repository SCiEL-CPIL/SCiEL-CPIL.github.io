"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    # Rank-35 decomposition using Strassen's algorithm on 5 independent 2x2 blocks.
    # Left operand A is 2x2. Right operand B is 2x10, viewed as 5 blocks of 2x2.
    # For each block, we apply Strassen's 7 products. Total rank = 5 * 7 = 35.
    #
    # Strassen's algorithm for 2x2 matrix multiplication C = A * B:
    #   M1 = (a11 + a22) * (b11 + b22)
    #   M2 = (a21 + a22) * b11
    #   M3 = a11 * (b12 - b22)
    #   M4 = a22 * (b21 - b11)
    #   M5 = (a11 + a12) * b22
    #   M6 = (a21 - a11) * (b11 + b12)
    #   M7 = (a12 - a22) * (b21 + b22)
    #
    #   c11 = M1 + M4 - M5 - M7
    #   c12 = M3 + M5
    #   c21 = M2 + M4
    #   c22 = M1 - M2 + M3 + M6
    #
    # Here A is fixed (2x2), and each block B^(q) is 2x2 with columns 2q-1, 2q.
    #
    # Indexing convention:
    #   U: shape (n*m, rank) = (4, 35), row index i*m + j (0-indexed)
    #   V: shape (m*p, rank) = (20, 35), row index j*p + k (0-indexed)
    #   W: shape (p*n, rank) = (20, 35), row index k*n + i (0-indexed), trace convention
    #
    # The tensor entry T[i, j, k] = 1 if C[i,k] = sum_{j} A[i,j] * B[j,k]
    # In the decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # For each Strassen product M_s within block q, we have one rank-1 term:
    #   U contribution: coefficients of the left linear form (linear in A entries)
    #   V contribution: coefficients of the right linear form (linear in B entries for block q)
    #   W contribution: coefficients mapping the product M_s to C entries (for block q)

    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    rank = 35
    U = [[0] * rank for _ in range(n * m)]   # 4 x 35
    V = [[0] * rank for _ in range(m * p)]   # 20 x 35
    W = [[0] * rank for _ in range(p * n)]   # 20 x 35

    # Strassen left linear forms (coefficients for a00, a10, a01, a11 in 0-indexed)
    # A is 2x2: A[0,0]=a00, A[0,1]=a01, A[1,0]=a10, A[1,1]=a11
    # U row index: i*m + j where i is row of A, j is col of A
    # So U row 0 = a00, row 1 = a01, row 2 = a10, row 3 = a11
    
    # Strassen right linear forms for block q (coefficients for b00, b01, b10, b11 in 0-indexed)
    # B block q has columns 2q, 2q+1 (0-indexed)
    # B[0, 2q] = b00, B[0, 2q+1] = b01, B[1, 2q] = b10, B[1, 2q+1] = b11
    # V row index: j*p + k where j is row of B, k is col of B
    # So V row 2q = b00, row 2q+1 = b01, row 10+2q = b10, row 10+2q+1 = b11
    
    # W maps product M_s to C entries for block q
    # C is 2x10: C[0, 2q], C[0, 2q+1], C[1, 2q], C[1, 2q+1]
    # W row index: k*n + i where k is col of C, i is row of C
    # So W row 2q = C[0, 2q], W row 2q+1 = C[0, 2q+1], W row 2+2q = C[1, 2q], W row 3+2q = C[1, 2q+1]

    # Strassen parameters: for each of the 7 products
    # left_coeffs: (coeff_a00, coeff_a01, coeff_a10, coeff_a11)
    # right_coeffs: (coeff_b00, coeff_b01, coeff_b10, coeff_b11)
    # c_contrib: (contrib_to_c00, contrib_to_c01, contrib_to_c10, contrib_to_c11)
    
    strassen_params = [
        # M1 = (a00 + a11) * (b00 + b11)
        ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)),
        # M2 = (a10 + a11) * b00
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, 0)),
        # M3 = a00 * (b01 - b11)
        ((1, 0, 0, 0), (0, 1, 0, -1), (0, 1, 0, 0)),
        # M4 = a11 * (b10 - b00)
        ((0, 0, 0, 1), (-1, 0, 1, 0), (1, 0, 1, 0)),
        # M5 = (a00 + a01) * b11
        ((1, 1, 0, 0), (0, 0, 0, 1), (-1, 1, 0, 0)),
        # M6 = (a10 - a00) * (b00 + b01)
        ((-1, 0, 1, 0), (1, 1, 0, 0), (0, 0, 0, 1)),
        # M7 = (a01 - a11) * (b10 + b11)
        ((0, 1, 0, -1), (0, 0, 1, 1), (-1, 0, 0, 0)),
    ]

    for q in range(5):
        for s in range(7):
            col = q * 7 + s
            left_c, right_c, c_c = strassen_params[s]
            
            # U: fill left coefficients
            U[0][col] += left_c[0]  # a00
            U[1][col] += left_c[1]  # a01
            U[2][col] += left_c[2]  # a10
            U[3][col] += left_c[3]  # a11
            
            # V: fill right coefficients for block q
            V[0 * p + (2 * q) + 0][col] += right_c[0]    # b00 of block q
            V[0 * p + (2 * q) + 1][col] += right_c[1]    # b01 of block q
            V[1 * p + (2 * q) + 0][col] += right_c[2]    # b10 of block q
            V[1 * p + (2 * q) + 1][col] += right_c[3]    # b11 of block q
            
            # W: fill contribution to C entries for block q
            # C[0, 2q] row in W: k*n + i = (2q)*2 + 0 = 4q
            W[(2 * q) * n + 0][col] += c_c[0]   # c00
            # C[0, 2q+1] row in W: k*n + i = (2q+1)*2 + 0 = 4q + 2
            W[(2 * q + 1) * n + 0][col] += c_c[1]  # c01
            # C[1, 2q] row in W: k*n + i = (2q)*2 + 1 = 4q + 1
            W[(2 * q) * n + 1][col] += c_c[2]     # c10
            # C[1, 2q+1] row in W: k*n + i = (2q+1)*2 + 1 = 4q + 3
            W[(2 * q + 1) * n + 1][col] += c_c[3]  # c11

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
