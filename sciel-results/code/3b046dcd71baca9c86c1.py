import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    rank = 50
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Outer Strassen coefficients for 4x4x4
    # alpha[k][I][J] is coeff of A[I][J] (2x2 block) in L_A[k]
    # beta[k][J][K] is coeff of B[J][K] (2x2 block) in L_B[k]
    # gamma[k][P][Q] is coeff of C[P][Q] (2x2 block) in sum of M_k
    
    alpha = [
        [[1, 0], [0, 1]],  # M1: A00 + A11
        [[0, 0], [1, 1]],  # M2: A10 + A11
        [[1, 0], [0, 0]],  # M3: A00
        [[0, 0], [0, 1]],  # M4: A11
        [[1, 1], [0, 0]],  # M5: A00 + A01
        [[-1, 0], [1, 0]], # M6: A10 - A00
        [[0, 1], [0, -1]]  # M7: A01 - A11
    ]
    
    beta = [
        [[1, 0], [0, 1]],  # M1: B00 + B11
        [[1, 0], [0, 0]],  # M2: B00
        [[0, 1], [0, -1]], # M3: B01 - B11
        [[0, -1], [1, 0]], # M4: B10 - B00
        [[0, 0], [0, 1]],  # M5: B11
        [[1, 1], [0, 0]],  # M6: B00 + B01
        [[0, 1], [1, 1]]   # M7: B10 + B11
    ]
    
    gamma = [
        [[1, 0], [0, 1]],  # M1: C00 + C11
        [[0, 0], [1, 0]],  # M2: C10
        [[0, 1], [0, 0]],  # M3: C01
        [[0, 0], [0, -1]], # M4: -C11 ? No. C00 -= M4, C01 += M4.
                           # Let's re-verify gamma.
                           # C00 = M1 - M4 + M5 + M7
                           # C01 = M3 + M4
                           # C10 = M2 + M5
                           # C11 = M1 - M2 + M3 + M6
                           
        # So:
        # M1: C00 += 1, C11 += 1
        # M2: C10 += 1, C11 -= 1
        # M3: C01 += 1, C11 += 1
        # M4: C00 -= 1, C01 += 1
        # M5: C00 += 1, C10 += 1
        # M6: C11 += 1
        # M7: C00 += 1
    ]
    
    # Correct gamma
    gamma = [
        [[1, 0], [0, 1]],  # M1
        [[0, 0], [1, -1]], # M2
        [[0, 1], [0, 1]],  # M3
        [[-1, 1], [0, 0]], # M4
        [[1, 0], [1, 0]],  # M5
        [[0, 0], [0, 1]],  # M6
        [[1, 0], [0, 0]]   # M7
    ]

    # Inner Strassen factors for 2x2x2
    # u_r, v_r, w_r are 2x2 matrices
    # r=0: M1
    u_inner = [
        [[1, 0], [0, 1]], # 1
        [[0, 0], [1, 1]], # 2
        [[1, 0], [0, 0]], # 3
        [[0, 0], [0, 1]], # 4
        [[1, 1], [0, 0]], # 5
        [[-1, 0], [1, 0]],# 6
        [[0, 1], [0, -1]] # 7
    ]
    
    v_inner = [
        [[1, 0], [0, 1]], # 1
        [[1, 0], [0, 0]], # 2
        [[0, 1], [0, -1]],# 3
        [[0, -1], [1, 0]],# 4
        [[0, 0], [0, 1]], # 5
        [[1, 1], [0, 0]], # 6
        [[0, 1], [1, 1]]  # 7
    ]
    
    w_inner = [
        [[1, 0], [0, 1]], # 1
        [[0, 0], [1, 0]], # 2
        [[0, 1], [0, 0]], # 3
        [[0, 0], [0, -1]],# 4
        [[1, 0], [1, 0]], # 5
        [[0, 0], [0, 1]], # 6
        [[1, 0], [0, 0]]  # 7
    ]
    
    # Fill U, V, W for the 49 terms
    col = 0
    for k in range(7):
        for r in range(7):
            # Loop over local block indices I, J, K (0 or 1)
            # and local element indices x, y, z (0 or 1)
            for I in range(2):
                for J in range(2):
                    for K in range(2):
                        a_coeff = alpha[k][I][J]
                        b_coeff = beta[k][J][K]
                        c_coeff = gamma[k][I][K] # Note: C block index is (I, K) ? 
                                                 # C = A B. A is 4x4, B is 4x4.
                                                 # C[I][K] is the block at row I, col K.
                                                 # In Strassen, C00 is top-left.
                                                 # My gamma[k][P][Q] corresponds to C[P][Q].
                                                 # So c_coeff = gamma[k][I][K] is correct.
                                                 # Wait, gamma is defined for C blocks.
                                                 # C00 is I=0, K=0.
                                                 # C11 is I=1, K=1.
                                                 # So yes, gamma[k][I][K].
                        
                        if a_coeff == 0 or b_coeff == 0 or c_coeff == 0:
                            continue
                            
                        for x in range(2):
                            for y in range(2):
                                for z in range(2):
                                    u_val = u_inner[r][x][y]
                                    v_val = v_inner[r][y][z]
                                    w_val = w_inner[r][z][x]
                                    
                                    if u_val == 0 or v_val == 0 or w_val == 0:
                                        continue
                                        
                                    val = a_coeff * b_coeff * c_coeff * u_val * v_val * w_val
                                    
                                    # Global indices
                                    i_global = 2 * I + x
                                    j_global = 2 * J + y
                                    k_global = 2 * K + z
                                    
                                    U[i_global * m + j_global][col] += val
                                    V[j_global * p + k_global][col] += val
                                    W[k_global * n + i_global][col] += val
            col += 1
            
    # Add the 1 term for j=4 (the extra column/row)
    # T[i, 4, k] = 1
    # U[i, 4, col] = 1
    # V[4, k, col] = 1
    # W[k, i, col] = 1
    for i in range(n):
        U[i * m + 4][col] = 1.0
    for k in range(p):
        V[4 * p + k][col] = 1.0
    for k in range(p):
        for i in range(n):
            W[k * n + i][col] = 1.0
    col += 1

    return {"U": U, "V": V, "W": W}
