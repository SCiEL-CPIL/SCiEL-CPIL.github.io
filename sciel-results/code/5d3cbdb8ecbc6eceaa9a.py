"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen products definition
    # P1: (a00+a11)*(b00+b11)
    # P2: (a10+a11)*b00
    # P3: a00*(b01-b11)
    # P4: a11*(b10-b00)
    # P5: (a00+a01)*b11
    # P6: (a10-a00)*(b00+b01)
    # P7: (a01-a11)*(b10+b11)
    
    strassen_products = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)]),
        ([(0, 1)], [(1, 1), (3, -1)]),
        ([(3, 1)], [(2, 1), (0, -1)]),
        ([(0, 1), (1, 1)], [(3, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),
    ]

    c_reconstruction = [
        [(0, 1), (3, 1), (4, -1), (6, 1)],
        [(2, 1), (4, 1)],
        [(1, 1), (3, 1)],
        [(0, 1), (1, -1), (2, 1), (5, 1)],
    ]

    # 8 Strassen instances
    # (a_col_off, b_row_off, b_col_off)
    instances = [
        (0, 0, 0), (0, 0, 2), (2, 2, 0), (2, 2, 2),
        (0, 0, 4), (0, 0, 6), (2, 2, 4), (2, 2, 6)
    ]

    col = 0
    for a_col_off, b_row_off, b_col_off in instances:
        for prod_idx in range(7):
            a_factors, b_factors = strassen_products[prod_idx]
            for a_elem, a_coeff in a_factors:
                a_row = a_elem // 2
                a_col = a_elem % 2
                u_idx = a_row * m + (a_col_off + a_col)
                U[u_idx][col] += a_coeff
            for b_elem, b_coeff in b_factors:
                b_row = b_elem // 2
                b_col = b_elem % 2
                v_idx = (b_row_off + b_row) * p + (b_col_off + b_col)
                V[v_idx][col] += b_coeff
            col += 1

    col = 0
    for a_col_off, b_row_off, b_col_off in instances:
        for c_elem in range(4):
            c_row = c_elem // 2
            c_col = c_elem % 2
            w_idx = (b_col_off + c_col) * n + c_row
            for prod_idx, coeff in c_reconstruction[c_elem]:
                W[w_idx][col + prod_idx] += coeff
        col += 7

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
