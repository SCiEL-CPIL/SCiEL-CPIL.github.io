"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set U, V, W entries
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    def add_term(col, u_idx, v_idx, w_idx, coeff_u, coeff_v, coeff_w):
        U[u_idx][col] += coeff_u
        V[v_idx][col] += coeff_v
        W[w_idx][col] += coeff_w

    # --- Strassen Block 1: A[0:2, 0:2] * B[0:2, 0:2] ---
    # A indices: 00(0), 01(1), 10(3), 11(4)
    # B indices: 00(0), 01(1), 10(5), 11(6)
    
    # M1: (A00+A11)(B00+B11)
    add_term(0, 0, 0, 0, 1, 1, 1) # C00
    add_term(0, 0, 0, 3, 1, 1, 1) # C11
    add_term(0, 3, 0, 0, 1, 1, 1) # C00
    add_term(0, 3, 0, 3, 1, 1, 1) # C11
    add_term(0, 0, 5, 0, 1, 1, 1) # C00
    add_term(0, 0, 5, 3, 1, 1, 1) # C11
    add_term(0, 3, 5, 0, 1, 1, 1) # C00
    add_term(0, 3, 5, 3, 1, 1, 1) # C11
    
    # M2: (A10+A11)B00
    add_term(1, 3, 0, 1, 1, 1, 1) # C10
    add_term(1, 3, 0, 3, 1, 1, -1) # C11
    add_term(1, 4, 0, 1, 1, 1, 1) # C10
    add_term(1, 4, 0, 3, 1, 1, -1) # C11
    
    # M3: A00(B01-B11)
    add_term(2, 0, 1, 2, 1, 1, 1) # C01
    add_term(2, 0, 1, 3, 1, 1, 1) # C11
    add_term(2, 0, 6, 2, 1, -1, 1) # C01
    add_term(2, 0, 6, 3, 1, -1, 1) # C11
    
    # M4: A11(B10-B01)
    add_term(3, 4, 5, 0, 1, 1, 1) # C00
    add_term(3, 4, 5, 1, 1, 1, 1) # C10
    add_term(3, 4, 1, 0, 1, -1, 1) # C00
    add_term(3, 4, 1, 1, 1, -1, 1) # C10
    
    # M5: (A00+A01)B11
    add_term(4, 0, 6, 0, 1, 1, -1) # C00
    add_term(4, 0, 6, 2, 1, 1, 1) # C01
    add_term(4, 1, 6, 0, 1, 1, -1) # C00
    add_term(4, 1, 6, 2, 1, 1, 1) # C01
    
    # M6: (A10-A00)(B00+B01)
    add_term(5, 3, 0, 3, 1, 1, 1) # C11
    add_term(5, 3, 1, 3, 1, 1, 1) # C11
    add_term(5, 0, 0, 3, -1, 1, 1) # C11
    add_term(5, 0, 1, 3, -1, 1, 1) # C11
    
    # M7: (A01-A11)(B10+B11)
    add_term(6, 1, 5, 0, 1, 1, 1) # C00
    add_term(6, 1, 6, 0, 1, 1, 1) # C00
    add_term(6, 4, 5, 0, -1, 1, 1) # C00
    add_term(6, 4, 6, 0, -1, 1, 1) # C00

    # --- Strassen Block 2: A[0:2, 0:2] * B[0:2, 2:4] ---
    # B indices: 02(2), 03(3), 12(7), 13(8)
    # C outputs: 02(4), 03(6), 12(5), 13(7)
    
    # M8: (A00+A11)(B02+B13)
    add_term(7, 0, 2, 4, 1, 1, 1) # C02
    add_term(7, 0, 2, 7, 1, 1, 1) # C13
    add_term(7, 3, 2, 4, 1, 1, 1) # C02
    add_term(7, 3, 2, 7, 1, 1, 1) # C13
    add_term(7, 0, 8, 4, 1, 1, 1) # C02
    add_term(7, 0, 8, 7, 1, 1, 1) # C13
    add_term(7, 3, 8, 4, 1, 1, 1) # C02
    add_term(7, 3, 8, 7, 1, 1, 1) # C13
    
    # M9: (A10+A11)B02
    add_term(8, 3, 2, 5, 1, 1, 1) # C12
    add_term(8, 3, 2, 7, 1, 1, -1) # C13
    add_term(8, 4, 2, 5, 1, 1, 1) # C12
    add_term(8, 4, 2, 7, 1, 1, -1) # C13
    
    # M10: A00(B03-B13)
    add_term(9, 0, 3, 6, 1, 1, 1) # C03
    add_term(9, 0, 3, 7, 1, 1, 1) # C13
    add_term(9, 0, 8, 6, 1, -1, 1) # C03
    add_term(9, 0, 8, 7, 1, -1, 1) # C13
    
    # M11: A11(B12-B03)
    add_term(10, 4, 7, 4, 1, 1, 1) # C02
    add_term(10, 4, 7, 5, 1, 1, 1) # C12
    add_term(10, 4, 3, 4, 1, -1, 1) # C02
    add_term(10, 4, 3, 5, 1, -1, 1) # C12
    
    # M12: (A00+A01)B13
    add_term(11, 0, 8, 4, 1, 1, -1) # C02
    add_term(11, 0, 8, 6, 1, 1, 1) # C03
    add_term(11, 1, 8, 4, 1, 1, -1) # C02
    add_term(11, 1, 8, 6, 1, 1, 1) # C03
    
    # M13: (A10-A00)(B02+B03)
    add_term(12, 3, 2, 7, 1, 1, 1) # C13
    add_term(12, 3, 3, 7, 1, 1, 1) # C13
    add_term(12, 0, 2, 7, -1, 1, 1) # C13
    add_term(12, 0, 3, 7, -1, 1, 1) # C13
    
    # M14: (A01-A11)(B12+B13)
    add_term(13, 1, 7, 4, 1, 1, 1) # C02
    add_term(13, 1, 8, 4, 1, 1, 1) # C02
    add_term(13, 4, 7, 4, -1, 1, 1) # C02
    add_term(13, 4, 8, 4, -1, 1, 1) # C02

    # --- Schoolbook Block 3: A[0:2, 0:2] * B[0:2, 4] ---
    # B indices: 04(4), 14(9)
    # C outputs: 04(8), 14(9)
    
    # M15: A00 B04
    add_term(14, 0, 4, 8, 1, 1, 1) # C04
    
    # M16: A01 B14
    add_term(15, 1, 9, 8, 1, 1, 1) # C04
    
    # M17: A10 B04
    add_term(16, 3, 4, 9, 1, 1, 1) # C14
    
    # M18: A11 B14
    add_term(17, 4, 9, 9, 1, 1, 1) # C14

    # --- Schoolbook Block 4: A[0:2, 2] * B[2, 0:5] ---
    # A indices: 02(2), 12(5)
    # B indices: 20(10), 21(11), 22(12), 23(13), 24(14)
    
    # M19: A02 B20 -> C00 (W idx 0)
    add_term(18, 2, 10, 0, 1, 1, 1)
    
    # M20: A02 B21 -> C01 (W idx 2)
    add_term(19, 2, 11, 2, 1, 1, 1)
    
    # M21: A02 B22 -> C02 (W idx 4)
    add_term(20, 2, 12, 4, 1, 1, 1)
    
    # M22: A02 B23 -> C03 (W idx 6)
    add_term(21, 2, 13, 6, 1, 1, 1)
    
    # M23: A02 B24 -> C04 (W idx 8)
    add_term(22, 2, 14, 8, 1, 1, 1)
    
    # M24: A12 B20 -> C10 (W idx 1)
    add_term(23, 5, 10, 1, 1, 1, 1)
    
    # M25: A12 B21 -> C11 (W idx 3)
    add_term(24, 5, 11, 3, 1, 1, 1)
    
    # M26: A12 B22 -> C12 (W idx 5)
    add_term(25, 5, 12, 5, 1, 1, 1)
    
    # M27: A12 B23 -> C13 (W idx 7)
    add_term(26, 5, 13, 7, 1, 1, 1)
    
    # M28: A12 B24 -> C14 (W idx 9)
    add_term(27, 5, 14, 9, 1, 1, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
