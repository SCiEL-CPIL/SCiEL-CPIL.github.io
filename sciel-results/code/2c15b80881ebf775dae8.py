"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 6

    # Strassen rank-7 decomposition for 2x2 matrices.
    # A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
    # C = A*B where C[i,j] = sum_k A[i,k]B[k,j]
    # 
    # The 7 multiplications are:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B10 - B00)
    # M4 = A11 (B01 - B11)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 - M5 + M4 + M7
    # C10 = M2 + M6
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6

    # U vectors (for A) - flattened as [a00, a01, a10, a11]
    U_strassen = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [1, 0, -1, 0],  # M6
        [0, 1, 0, -1],  # M7
    ]

    # V vectors (for B) - flattened as [b00, b01, b10, b11]
    V_strassen = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 0, 1, -1],  # M3
        [0, 1, 0, -1],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1],   # M7
    ]

    # W vectors (for C) - flattened as [c00, c10, c01, c11] (trace convention: k*n+i)
    # For 2x2, trace convention gives index = k*2 + i
    # c00: k=0, i=0 -> index 0
    # c10: k=0, i=1 -> index 1
    # c01: k=1, i=0 -> index 2
    # c11: k=1, i=1 -> index 3
    W_strassen = [
        [1, 0, 0, 1],   # M1: C00 += 1, C11 += 1
        [0, 1, 0, -1],  # M2: C10 += 1, C11 -= 1
        [0, 0, 1, 1],   # M3: C01 += 1, C11 += 1
        [1, 0, 0, 0],   # M4: C00 += 1
        [-1, 0, 1, 0],  # M5: C00 -= 1, C01 += 1
        [0, 1, 0, 1],   # M6: C10 += 1, C11 += 1
        [1, 0, 0, 0],   # M7: C00 += 1
    ]

    # Let's verify Strassen correctness:
    # C00 = M1 - M5 + M4 + M7
    # W contributions to C00 (index 0): 1 + (-1) + 1 + 1 = 2? No, that's wrong.
    # Wait, let me re-check.
    # M1: [1, 0, 0, 1] -> C00 += 1
    # M4: [1, 0, 0, 0] -> C00 += 1
    # M5: [-1, 0, 1, 0] -> C00 -= 1
    # M7: [1, 0, 0, 0] -> C00 += 1
    # Sum for C00: 1 + 1 - 1 + 1 = 2. But it should be 1.
    # 
    # I think I have the Strassen formulas wrong. Let me look up the correct ones.
    # 
    # Correct Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M5 + M4 + M7
    # C10 = M2 + M6
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    #
    # Let's verify C00 = A00B00 + A01B10:
    # M1 = A00B00 + A00B11 + A11B00 + A11B11
    # -M5 = -A00B11 - A01B11
    # M4 = A11B01 - A11B11
    # M7 = A01B10 + A01B11 - A11B10 - A11B11
    # Sum = A00B00 + A00B11 + A11B00 + A11B11 - A00B11 - A01B11 + A11B01 - A11B11 + A01B10 + A01B11 - A11B10 - A11B11
    # = A00B00 + A11B00 + A11B01 + A01B10 - A11B10 - A11B11
    # This is NOT C00 = A00B00 + A01B10.
    #
    # I must have the Strassen formulas wrong. Let me use a different approach.
    # Let me just implement the schoolbook decomposition for 4x4x6, which has rank 96.
    # This is a valid decomposition, even if not optimal.

    rank = 4 * 4 * 6
    U = [[0] * rank for _ in range(4 * 4)]
    V = [[0] * rank for _ in range(4 * 6)]
    W = [[0] * rank for _ in range(6 * 4)]
    column = 0
    for i in range(4):
        for j in range(4):
            for k in range(6):
                U[i * 4 + j][column] = 1
                V[j * 6 + k][column] = 1
                W[k * 4 + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
