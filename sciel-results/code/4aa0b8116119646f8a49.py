import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 140s limit, 10s margin

    n, m, p = DIMENSIONS
    # 8 blocks: 2 in j-dimension, 4 in k-dimension
    # Each block is 2x2x2 Strassen (rank 7)
    # Total rank = 8 * 7 = 56
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56

    # Strassen coefficients for 2x2 matrices
    # Indices: a (row A/C), c (col A / row B), d (col B/C)
    # U_coeffs[s][a][c]
    # V_coeffs[s][c][d]
    # W_coeffs[s][a][d]
    
    # s=0: M1
    U_c = [[[0,0],[0,0]], [[0,0],[1,0]], [[1,0],[0,0]], [[0,0],[0,0]], [[1,0],[1,0]], [[-1,0],[0,0]], [[0,1],[-1,0]]]
    # Wait, let's build them explicitly to avoid errors.
    
    U_coeffs = [
        [[1, 0], [0, 1]], # M1: A00, A11
        [[0, 0], [1, 1]], # M2: A10, A11
        [[1, 0], [0, 0]], # M3: A00
        [[0, 0], [0, 1]], # M4: A11
        [[1, 0], [1, 0]], # M5: A00, A10
        [[-1, 0], [0, 1]],# M6: A11-A00
        [[0, 1], [-1, 0]] # M7: A01-A11
    ]
    
    V_coeffs = [
        [[1, 0], [0, 1]], # M1: B00, B11
        [[1, 0], [0, 0]], # M2: B00
        [[0, 0], [1, -1]],# M3: B10-B11
        [[-1, 1], [0, 0]],# M4: B01-B00
        [[0, 0], [0, 1]], # M5: B11
        [[1, 1], [0, 0]], # M6: B00+B01
        [[0, 0], [1, 1]]  # M7: B10+B11
    ]
    
    W_coeffs = [
        [[1, 0], [0, 1]], # M1: C00, C11
        [[0, 1], [1, 0]], # M2: C01, C10
        [[0, 1], [1, -1]],# M3: C01, C10, -C11
        [[1, 0], [1, 0]], # M4: C00, C10
        [[-1, 0], [0, 1]],# M5: -C00, C11
        [[-1, -1], [0, -1]],# M6: -C00, -C01, -C11
        [[1, 1], [0, 0]]  # M7: C00, C01
    ]

    # Verify W coeffs against formulas:
    # C00 = M1 + M4 - M5 + M7 -> 1 + 1 - (-1) + 1 = 4? No.
    # Formula: C00 = M1 + M4 - M5 + M7
    # My W_coeffs for C00 (index [0][0]):
    # M1: 1
    # M2: 0
    # M3: 0
    # M4: 1
    # M5: -1
    # M6: -1
    # M7: 1
    # Sum: 1 + 1 - 1 + 1 = 2?
    # Let's re-read Strassen output formulas.
    # C00 = M1 + M4 - M5 + M7
    # If I sum the W coefficients for C00 over all M's, I should get the coefficient of C00 in the final sum?
    # No, W_coeffs[s][a][d] is the coefficient of P_s in C_{a,d}.
    # So for C00, the contribution from M1 is 1, from M4 is 1, from M5 is -1, from M7 is 1.
    # My table:
    # W_coeffs[0][0][0] = 1 (M1)
    # W_coeffs[3][0][0] = 1 (M4)
    # W_coeffs[4][0][0] = -1 (M5)
    # W_coeffs[6][0][0] = 1 (M7)
    # This matches.

    # C01 = M2 + M3 - M6 + M7
    # W_coeffs[1][0][1] = 1 (M2)
    # W_coeffs[2][0][1] = 1 (M3)
    # W_coeffs[5][0][1] = -1 (M6)
    # W_coeffs[6][0][1] = 1 (M7)
    # Matches.

    # C10 = M2 + M3 + M4
    # W_coeffs[1][1][0] = 1 (M2)
    # W_coeffs[2][1][0] = 1 (M3)
    # W_coeffs[3][1][0] = 1 (M4)
    # Matches.

    # C11 = M1 - M3 + M5 - M6
    # W_coeffs[0][1][1] = 1 (M1)
    # W_coeffs[2][1][1] = -1 (M3)
    # W_coeffs[4][1][1] = 1 (M5)
    # W_coeffs[5][1][1] = -1 (M6)
    # Matches.

    # Mapping local to global
    # Local a in {0,1} -> Global i = a
    # Local c in {0,1} -> Global j = 2*J_b + c
    # Local d in {0,1} -> Global k = 2*K_b + d
    
    # U index: i*m + j = 4*i + j
    # V index: j*p + k = 8*j + k
    # W index: k*n + i = 2*k + i

    for J_b in range(2):
        for K_b in range(4):
            block_id = J_b * 4 + K_b
            base_rank = block_id * 7
            
            for s in range(7):
                R = base_rank + s
                
                if time.time() > deadline:
                    break

                # Update U
                for a in range(2):
                    for c in range(2):
                        val = U_coeffs[s][a][c]
                        if val != 0:
                            i = a
                            j = 2 * J_b + c
                            U[i * m + j][R] += val

                # Update V
                for c in range(2):
                    for d in range(2):
                        val = V_coeffs[s][c][d]
                        if val != 0:
                            j = 2 * J_b + c
                            k = 2 * K_b + d
                            V[j * p + k][R] += val

                # Update W
                for a in range(2):
                    for d in range(2):
                        val = W_coeffs[s][a][d]
                        if val != 0:
                            i = a
                            k = 2 * K_b + d
                            W[k * n + i][R] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
