"""Matrix multiplication 2,2,6 - Strassen-based rank-21 decomposition."""

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 6
    
    # Strassen 2x2 algorithm: 7 multiplications
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)(b11)
    # M3 = (a11)(b21-b22)
    # M4 = (a22)(b12-b11)
    # M5 = (a11+a12)(b22)
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    # For each of 3 blocks of B (2x6 -> three 2x2 blocks):
    # Block t (t=0,1,2): B columns 2t, 2t+1
    # Output C columns 2t, 2t+1
    
    # Strassen A-factors (row of A, coefficient):
    # A indices: row 0 = a11, a12; row 1 = a21, a22
    # M1: a11+a22 -> rows (0,0):1, (1,1):1
    # M2: a21+a22 -> rows (1,0):1, (1,1):1
    # M3: a11 -> row (0,0):1
    # M4: a22 -> row (1,1):1
    # M5: a11+a12 -> rows (0,0):1, (0,1):1
    # M6: a21-a11 -> rows (1,0):1, (0,0):-1
    # M7: a12-a22 -> rows (0,1):1, (1,1):-1
    
    # Strassen B-factors for a block (col of B_block, coefficient):
    # B_block indices: row 0 = b11, b12; row 1 = b21, b22
    # M1: b11+b22 -> (0,0):1, (1,1):1
    # M2: b11 -> (0,0):1
    # M3: b21-b22 -> (1,0):1, (1,1):-1
    # M4: b12-b11 -> (0,1):1, (0,0):-1
    # M5: b22 -> (1,1):1
    # M6: b11+b12 -> (0,0):1, (0,1):1
    # M7: b21+b22 -> (1,0):1, (1,1):1
    
    # W-factors: output C = [[c11, c12], [c21, c22]]
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    # W index convention: W[k*n+i, r] where k=col of C, i=row of C
    # C is 2x6, so k in 0..5, i in 0..1
    # W index = k*n + i = k*2 + i
    
    # For block t, output columns are 2t and 2t+1
    # c11 (row 0, col 2t) -> W[2t*2+0, r] = W[4t, r]
    # c12 (row 0, col 2t+1) -> W[4t+1, r]
    # c21 (row 1, col 2t) -> W[4t+2, r]
    # c22 (row 1, col 2t+1) -> W[4t+3, r]
    
    # W coefficients per Strassen product:
    # M1: c11:+1, c22:+1
    # M2: c21:+1, c22:-1
    # M3: c12:+1, c22:+1
    # M4: c11:+1, c21:+1
    # M5: c11:-1, c12:+1
    # M6: c22:+1
    # M7: c11:+1
    
    rank = 21
    U = [[0] * rank for _ in range(n * m)]  # 4 x 21
    V = [[0] * rank for _ in range(m * p)]  # 12 x 21
    W = [[0] * rank for _ in range(p * n)]  # 12 x 21
    
    # A-factors for each Strassen product (row, col, coeff)
    # A is 2x2, index = i*m+j = i*2+j
    a_factors = [
        [(0, 0, 1), (1, 1, 1)],       # M1: a11+a22
        [(1, 0, 1), (1, 1, 1)],       # M2: a21+a22
        [(0, 0, 1)],                   # M3: a11
        [(1, 1, 1)],                   # M4: a22
        [(0, 0, 1), (0, 1, 1)],       # M5: a11+a12
        [(1, 0, 1), (0, 0, -1)],      # M6: a21-a11
        [(0, 1, 1), (1, 1, -1)],      # M7: a12-a22
    ]
    
    # B-factors for each Strassen product (row, col, coeff)
    # B is 2x2 per block, index = j*p+k where j is row of B, k is col
    # For block t, col offset is 2t
    b_factors = [
        [(0, 0, 1), (1, 1, 1)],       # M1: b11+b22
        [(0, 0, 1)],                   # M2: b11
        [(1, 0, 1), (1, 1, -1)],      # M3: b21-b22
        [(0, 1, 1), (0, 0, -1)],      # M4: b12-b11
        [(1, 1, 1)],                   # M5: b22
        [(0, 0, 1), (0, 1, 1)],       # M6: b11+b12
        [(1, 0, 1), (1, 1, 1)],       # M7: b21+b22
    ]
    
    # W-factors for each Strassen product (row_of_C, col_of_C, coeff)
    # C is 2x6, W index = k*n+i = k*2+i
    w_factors = [
        [(0, 0, 1), (1, 1, 1)],       # M1: c11:+1, c22:+1
        [(1, 0, 1), (1, 1, -1)],      # M2: c21:+1, c22:-1
        [(0, 1, 1), (1, 1, 1)],       # M3: c12:+1, c22:+1
        [(0, 0, 1), (1, 0, 1)],       # M4: c11:+1, c21:+1
        [(0, 0, -1), (0, 1, 1)],      # M5: c11:-1, c12:+1
        [(1, 1, 1)],                   # M6: c22:+1
        [(0, 0, 1)],                   # M7: c11:+1
    ]
    
    col = 0
    for t in range(3):  # 3 blocks
        for s in range(7):  # 7 Strassen products
            # Fill U
            for (i, j, coeff) in a_factors[s]:
                U[i * m + j][col] += coeff
            
            # Fill V (B block t, col offset 2t)
            for (j, k, coeff) in b_factors[s]:
                V[j * p + (2 * t + k)][col] += coeff
            
            # Fill W (C block t, col offset 2t)
            for (i, k, coeff) in w_factors[s]:
                W[(2 * t + k) * n + i][col] += coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
