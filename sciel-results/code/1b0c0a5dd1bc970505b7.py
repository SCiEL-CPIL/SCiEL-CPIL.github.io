"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    # Strassen terms: each is (u_coeffs, v_coeffs, w_coeffs)
    # u/v/w coeffs are lists of (local_idx, coefficient)
    # local_idx for A/B/C 2x2: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_terms = [
        # M1: (a00+a11)*(b00+b11), contributes +1 to c00, +1 to c11
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2: (a10+a11)*b00, contributes +1 to c10, -1 to c11
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3: a00*(b10-b11), contributes +1 to c01, +1 to c11
        ([(0, 1)], [(2, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4: a11*(b01-b00), contributes +1 to c00, +1 to c10
        ([(3, 1)], [(1, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5: (a00+a01)*b11, contributes -1 to c00, +1 to c01
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6: (a10-a00)*(b00+b01), contributes +1 to c11
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7: (a01-a11)*(b10+b11), contributes +1 to c00
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    col = 0
    for jg in range(2):  # j_group: 0 or 1
        j_offset = jg * 2  # 0 or 2
        for kb in range(4):  # k_block: 0,1,2,3
            k_offset = kb * 2  # 0,2,4,6
            for (u_coeffs, v_coeffs, w_coeffs) in strassen_terms:
                for (a_idx, coeff) in u_coeffs:
                    i = a_idx // 2
                    j = j_offset + a_idx % 2
                    U[i * m + j][col] += coeff
                for (b_idx, coeff) in v_coeffs:
                    j = j_offset + b_idx // 2
                    k = k_offset + b_idx % 2
                    V[j * p + k][col] += coeff
                for (c_idx, coeff) in w_coeffs:
                    i = c_idx // 2
                    k = k_offset + c_idx % 2
                    W[k * n + i][col] += coeff
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
