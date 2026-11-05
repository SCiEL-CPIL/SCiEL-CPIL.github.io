"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-189 decomposition of 6x6 matrix multiplication using:
    - View 6x6 as 3x3 blocks of 2x2 matrices
    - Schoolbook on the 3x3 block level (27 block products)
    - Strassen (rank 7) for each 2x2 block product
    Total rank: 27 * 7 = 189
    """
    n, m, p = DIMENSIONS  # all 6
    
    rank = 189
    U = [[0] * rank for _ in range(36)]  # 6*6 = 36
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    # Strassen 2x2 terms: (U_coeffs, V_coeffs)
    # 2x2 block: element (r,c) where r,c in {0,1}
    strassen_2x2 = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),      # M1 = (A11+A22)*(B11+B22)
        ({(1,0):1, (1,1):1}, {(0,0):1}),                  # M2 = (A21+A22)*B11
        ({(0,0):1}, {(0,1):1, (1,1):-1}),                 # M3 = A11*(B12-B22)
        ({(1,1):1}, {(1,0):1, (0,0):-1}),                 # M4 = A22*(B21-B11)
        ({(0,0):1, (0,1):1}, {(1,1):1}),                  # M5 = (A11+A12)*B22
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),        # M6 = (A21-A11)*(B11+B12)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),        # M7 = (A12-A22)*(B21+B22)
    ]
    
    # W coefficients: for each 2x2 result element (a,c) and Strassen term t
    # C(0,0) = M1 + M4 - M5 + M7
    # C(0,1) = M3 + M5
    # C(1,0) = M2 + M4
    # C(1,1) = M1 - M2 + M3 + M6
    W_2x2 = [
        [1,  0, 0, 1, -1, 0, 1],  # C(0,0)
        [0,  0, 1, 0,  1, 0, 0],  # C(0,1)
        [0,  1, 0, 1,  0, 0, 0],  # C(1,0)
        [1, -1, 1, 0,  0, 1, 0],  # C(1,1)
    ]
    
    col = 0
    for i in range(3):  # block row of A (3x3 block structure)
        for j in range(3):  # block col of A = block row of B
            for k in range(3):  # block col of B
                for t in range(7):  # 7 Strassen terms
                    U_coeffs = strassen_2x2[t][0]
                    V_coeffs = strassen_2x2[t][1]
                    
                    for (ua, ub), u_val in U_coeffs.items():
                        for (va, vb), v_val in V_coeffs.items():
                            # A element: (2*i + ua, 2*j + ub)
                            # B element: (2*j + va, 2*k + vb)
                            for a in range(2):  # row in 2x2 result
                                for c in range(2):  # col in 2x2 result
                                    w_val = W_2x2[a * 2 + c][t]
                                    if w_val != 0:
                                        # C element: (2*i + a, 2*k + c)
                                        # W index: k_idx * n + i_idx = (2*k+c)*6 + (2*i+a)
                                        U[(2*i + ua) * 6 + (2*j + ub)][col] += u_val
                                        V[(2*j + va) * 6 + (2*k + vb)][col] += v_val
                                        W[(2*k + c) * 6 + (2*i + a)][col] += w_val
                            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
