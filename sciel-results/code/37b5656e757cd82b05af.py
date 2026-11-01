"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    # Construct a rank-18 exact decomposition for 2x2 @ 2x5 multiplication.
    # Uses Strassen's algorithm for the first two 2x2 blocks (columns 0-1 and 2-3)
    # and schoolbook for the last column (column 4).
    n, m, p = DIMENSIONS
    rank = 18
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # Helper to add a term to a specific column
    def add_term(col, u_idx, v_idx, w_idx, val):
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val

    # --- Block 1: Columns 0,1 of B (indices 0,1,5,6 in V) -> Rows 0,1 of C (indices 0,1,2,3 in W) ---
    # Strassen 2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # We need to map these to linear combinations in U, V, W.
    # Each M_i corresponds to one column in U,V,W.
    
    # Col 0: M1 = (a00+a11)(b00+b11)
    # U: a00 (idx 0), a11 (idx 3) -> +1
    # V: b00 (idx 0), b11 (idx 6) -> +1
    # W: C00 (idx 0), C11 (idx 3) -> +1 for M1
    add_term(0, 0, 0, 0, 1)   # a00 * b00 -> C00
    add_term(0, 0, 6, 0, 1)   # a00 * b11 -> C00
    add_term(0, 3, 0, 0, 1)   # a11 * b00 -> C00
    add_term(0, 3, 6, 0, 1)   # a11 * b11 -> C00
    add_term(0, 0, 0, 3, 1)   # a00 * b00 -> C11
    add_term(0, 0, 6, 3, 1)   # a00 * b11 -> C11
    add_term(0, 3, 0, 3, 1)   # a11 * b00 -> C11
    add_term(0, 3, 6, 3, 1)   # a11 * b11 -> C11

    # Col 1: M2 = (a10+a11)b00
    # U: a10 (idx 2), a11 (idx 3) -> +1
    # V: b00 (idx 0) -> +1
    # W: C10 (idx 2) -> +1, C11 (idx 3) -> -1
    add_term(1, 2, 0, 2, 1)   # a10 * b00 -> C10
    add_term(1, 3, 0, 2, 1)   # a11 * b00 -> C10
    add_term(1, 2, 0, 3, -1)  # a10 * b00 -> -C11
    add_term(1, 3, 0, 3, -1)  # a11 * b00 -> -C11

    # Col 2: M3 = a00(b01-b11)
    # U: a00 (idx 0) -> +1
    # V: b01 (idx 1), b11 (idx 6) -> +1, -1
    # W: C01 (idx 1) -> +1, C11 (idx 3) -> +1
    add_term(2, 0, 1, 1, 1)   # a00 * b01 -> C01
    add_term(2, 0, 6, 1, -1)  # a00 * b11 -> -C01
    add_term(2, 0, 1, 3, 1)   # a00 * b01 -> C11
    add_term(2, 0, 6, 3, -1)  # a00 * b11 -> -C11

    # Col 3: M4 = a11(b10-b00)
    # U: a11 (idx 3) -> +1
    # V: b10 (idx 5), b00 (idx 0) -> +1, -1
    # W: C00 (idx 0) -> +1, C10 (idx 2) -> +1
    add_term(3, 3, 5, 0, 1)   # a11 * b10 -> C00
    add_term(3, 3, 0, 0, -1)  # a11 * b00 -> -C00
    add_term(3, 3, 5, 2, 1)   # a11 * b10 -> C10
    add_term(3, 3, 0, 2, -1)  # a11 * b00 -> -C10

    # Col 4: M5 = (a00+a01)b11
    # U: a00 (idx 0), a01 (idx 1) -> +1
    # V: b11 (idx 6) -> +1
    # W: C00 (idx 0) -> -1, C01 (idx 1) -> +1
    add_term(4, 0, 6, 0, -1)  # a00 * b11 -> -C00
    add_term(4, 1, 6, 0, -1)  # a01 * b11 -> -C00
    add_term(4, 0, 6, 1, 1)   # a00 * b11 -> C01
    add_term(4, 1, 6, 1, 1)   # a01 * b11 -> C01

    # Col 5: M6 = (a10-a00)(b00+b01)
    # U: a10 (idx 2), a00 (idx 0) -> +1, -1
    # V: b00 (idx 0), b01 (idx 1) -> +1
    # W: C11 (idx 3) -> +1
    add_term(5, 2, 0, 3, 1)   # a10 * b00 -> C11
    add_term(5, 2, 1, 3, 1)   # a10 * b01 -> C11
    add_term(5, 0, 0, 3, -1)  # a00 * b00 -> -C11
    add_term(5, 0, 1, 3, -1)  # a00 * b01 -> -C11

    # Col 6: M7 = (a01-a11)(b10+b11)
    # U: a01 (idx 1), a11 (idx 3) -> +1, -1
    # V: b10 (idx 5), b11 (idx 6) -> +1
    # W: C00 (idx 0) -> +1
    add_term(6, 1, 5, 0, 1)   # a01 * b10 -> C00
    add_term(6, 1, 6, 0, 1)   # a01 * b11 -> C00
    add_term(6, 3, 5, 0, -1)  # a11 * b10 -> -C00
    add_term(6, 3, 6, 0, -1)  # a11 * b11 -> -C00

    # --- Block 2: Columns 2,3 of B (indices 2,3,7,8 in V) -> Rows 2,3 of C (indices 4,5,6,7 in W) ---
    # Same Strassen structure, but B indices shifted by +2, W indices shifted by +4.
    # V indices: b02=2, b03=3, b12=7, b13=8
    # W indices: C02=4, C03=5, C12=6, C13=7

    # Col 7: M8 = (a00+a11)(b02+b13)
    add_term(7, 0, 2, 4, 1)
    add_term(7, 0, 8, 4, 1)
    add_term(7, 3, 2, 4, 1)
    add_term(7, 3, 8, 4, 1)
    add_term(7, 0, 2, 7, 1)
    add_term(7, 0, 8, 7, 1)
    add_term(7, 3, 2, 7, 1)
    add_term(7, 3, 8, 7, 1)

    # Col 8: M9 = (a10+a11)b02
    add_term(8, 2, 2, 6, 1)
    add_term(8, 3, 2, 6, 1)
    add_term(8, 2, 2, 7, -1)
    add_term(8, 3, 2, 7, -1)

    # Col 9: M10 = a00(b03-b13)
    add_term(9, 0, 3, 5, 1)
    add_term(9, 0, 8, 5, -1)
    add_term(9, 0, 3, 7, 1)
    add_term(9, 0, 8, 7, -1)

    # Col 10: M11 = a11(b12-b02)
    add_term(10, 3, 7, 4, 1)
    add_term(10, 3, 2, 4, -1)
    add_term(10, 3, 7, 6, 1)
    add_term(10, 3, 2, 6, -1)

    # Col 11: M12 = (a00+a01)b13
    add_term(11, 0, 8, 4, -1)
    add_term(11, 1, 8, 4, -1)
    add_term(11, 0, 8, 5, 1)
    add_term(11, 1, 8, 5, 1)

    # Col 12: M13 = (a10-a00)(b02+b03)
    add_term(12, 2, 2, 7, 1)
    add_term(12, 2, 3, 7, 1)
    add_term(12, 0, 2, 7, -1)
    add_term(12, 0, 3, 7, -1)

    # Col 13: M14 = (a01-a11)(b12+b13)
    add_term(13, 1, 7, 4, 1)
    add_term(13, 1, 8, 4, 1)
    add_term(13, 3, 7, 4, -1)
    add_term(13, 3, 8, 4, -1)

    # --- Block 3: Column 4 of B (indices 4,9 in V) -> Row 4 of C (indices 8,9 in W) ---
    # Schoolbook for 2x1
    # C04 = a00*b04 + a01*b14
    # C14 = a10*b04 + a11*b14

    # Col 14: a00*b04
    add_term(14, 0, 4, 8, 1)

    # Col 15: a01*b14
    add_term(15, 1, 9, 8, 1)

    # Col 16: a10*b04
    add_term(16, 2, 4, 9, 1)

    # Col 17: a11*b14
    add_term(17, 3, 9, 9, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
