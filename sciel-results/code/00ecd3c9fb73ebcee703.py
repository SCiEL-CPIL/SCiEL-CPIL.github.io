DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    # We partition the 2x4 * 4x8 multiplication into 8 independent 2x2 * 2x2 problems.
    # u in {0,1} corresponds to the j-group (j = 2u, 2u+1)
    # w in {0,1,2,3} corresponds to the k-group (k = 2w, 2w+1)
    # For each (u,w), we apply the rank-7 Strassen decomposition.
    
    # Strassen indices for 2x2 matrices:
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    #
    # The 7 Strassen products and their contributions to U, V, W:
    # Each product is of the form: (sum of A entries) * (sum of B entries)
    # U vector: coefficients for A entries
    # V vector: coefficients for B entries
    # W vector: coefficients for C entries (which map to C[i, k] where i is row of A, k is col of B)
    
    # For each (u, w), the local 2x2 A sub-matrix entries map to global A indices:
    # a00 -> A[0, 2u], a01 -> A[0, 2u+1]
    # a10 -> A[1, 2u], a11 -> A[1, 2u+1]
    # Local B entries map to global B indices:
    # b00 -> B[2u, 2w], b01 -> B[2u, 2w+1]
    # b10 -> B[2u+1, 2w], b11 -> B[2u+1, 2w+1]
    # Local C entries map to global C indices:
    # c00 -> C[0, 2w], c01 -> C[0, 2w+1]
    # c10 -> C[1, 2w], c11 -> C[1, 2w+1]
    
    # Strassen formulas:
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    
    # C00 = M1 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So W contributions for each M:
    # M1: c00: +1, c11: +1
    # M2: c10: +1, c11: -1
    # M3: c01: +1, c11: +1
    # M4: c00: +1, c10: +1
    # M5: c00: -1, c01: +1
    # M6: c11: +1
    # M7: c00: +1
    
    # U contributions (coefficients on a00, a01, a10, a11):
    # M1: a00: 1, a11: 1
    # M2: a10: 1, a11: 1
    # M3: a00: 1
    # M4: a11: 1
    # M5: a00: 1, a01: 1
    # M6: a10: 1, a00: -1
    # M7: a01: 1, a11: -1
    
    # V contributions (coefficients on b00, b01, b10, b11):
    # M1: b00: 1, b11: 1
    # M2: b00: 1
    # M3: b01: 1, b11: -1
    # M4: b10: 1, b00: -1
    # M5: b11: 1
    # M6: b00: 1, b01: 1
    # M7: b10: 1, b11: 1
    
    # Define the 7 Strassen patterns as (U_coeffs, V_coeffs, W_coeffs)
    # where indices are: U: [a00, a01, a10, a11], V: [b00, b01, b10, b11], W: [c00, c01, c10, c11]
    
    strassen_patterns = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    rank = 7 * 2 * 4  # 7 patterns * 2 u-groups * 4 w-groups = 56
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    col = 0
    for u in range(2):
        for w in range(4):
            for (u_coeffs, v_coeffs, w_coeffs) in strassen_patterns:
                # U contributions: for each local A entry, add to global U row
                # a00 -> A[0, 2u] -> row 0*4 + 2u = 2u
                # a01 -> A[0, 2u+1] -> row 0*4 + 2u+1 = 2u+1
                # a10 -> A[1, 2u] -> row 1*4 + 2u = 4 + 2u
                # a11 -> A[1, 2u+1] -> row 1*4 + 2u+1 = 5 + 2u
                U[2*u][col] += u_coeffs[0]
                U[2*u + 1][col] += u_coeffs[1]
                U[4 + 2*u][col] += u_coeffs[2]
                U[5 + 2*u][col] += u_coeffs[3]
                
                # V contributions: for each local B entry, add to global V row
                # b00 -> B[2u, 2w] -> row 2u*8 + 2w
                # b01 -> B[2u, 2w+1] -> row 2u*8 + 2w+1
                # b10 -> B[2u+1, 2w] -> row (2u+1)*8 + 2w
                # b11 -> B[2u+1, 2w+1] -> row (2u+1)*8 + 2w+1
                V[2*u * 8 + 2*w][col] += v_coeffs[0]
                V[2*u * 8 + 2*w + 1][col] += v_coeffs[1]
                V[(2*u + 1) * 8 + 2*w][col] += v_coeffs[2]
                V[(2*u + 1) * 8 + 2*w + 1][col] += v_coeffs[3]
                
                # W contributions: for each local C entry, add to global W row
                # W index: k*n + i = k*2 + i
                # c00 -> C[0, 2w] -> row 2w*2 + 0 = 4w
                # c01 -> C[0, 2w+1] -> row (2w+1)*2 + 0 = 4w+2
                # c10 -> C[1, 2w] -> row 2w*2 + 1 = 4w+1
                # c11 -> C[1, 2w+1] -> row (2w+1)*2 + 1 = 4w+3
                W[4*w][col] += w_coeffs[0]
                W[4*w + 2][col] += w_coeffs[1]
                W[4*w + 1][col] += w_coeffs[2]
                W[4*w + 3][col] += w_coeffs[3]
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
