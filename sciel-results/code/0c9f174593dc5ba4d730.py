"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 33 decomposition
    # Split j=0,1 (Strassen 2x2x2 blocks) and j=2 (Schoolbook)
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j=2 (Schoolbook for 2x1x6)
    # A[i,2] * B[2,k] -> C[i,k]
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: j=0,1 (Strassen for 2x2x2 blocks, repeated 3 times for k pairs)
    # Strassen coefficients for 2x2x2
    # Indices: local i in {0,1}, j in {0,1}, k in {0,1}
    # U: (i,j), V: (j,k), W: (k,i)
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    # U: A00:1, A11:1
    # V: B00:1, B11:1
    # W: C00:1, C11:1
    # Term 2: M2 = (A10+A11)B00
    # U: A10:1, A11:1
    # V: B00:1
    # W: C10:1, C11:-1
    # Term 3: M3 = A00(B01-B11)
    # U: A00:1
    # V: B01:1, B11:-1
    # W: C01:1, C11:1
    # Term 4: M4 = A11(B10-B00)
    # U: A11:1
    # V: B10:1, B00:-1
    # W: C00:1, C10:1
    # Term 5: M5 = (A00+A01)B11
    # U: A00:1, A01:1
    # V: B11:1
    # W: C00:-1, C01:1
    # Term 6: M6 = (A10-A00)(B00+B01)
    # U: A10:1, A00:-1
    # V: B00:1, B01:1
    # W: C11:1
    # Term 7: M7 = (A01-A11)(B10+B11)
    # U: A01:1, A11:-1
    # V: B10:1, B11:1
    # W: C00:1

    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U: dict (i,j)->val, V: dict (j,k)->val, W: dict (k,i)->val
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}), # M1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}), # M2
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}), # M3
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}), # M4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}), # M5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}), # M6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}), # M7
    ]

    # 3 blocks for k: [0,1], [2,3], [4,5]
    for g in range(3):
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for (i, j), val in u_coeffs.items():
                U[i * m + j][col] += val
            for (j, k_local), val in v_coeffs.items():
                k_global = 2 * g + k_local
                V[j * p + k_global][col] += val
            for (k_local, i), val in w_coeffs.items():
                k_global = 2 * g + k_local
                W[k_global * n + i][col] += val
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
