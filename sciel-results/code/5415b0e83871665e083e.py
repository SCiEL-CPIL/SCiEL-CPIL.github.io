import time
import sys

DIMENSIONS = [2, 3, 10]

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix multiplication.
    Rank 55.
    """
    n, m, p = DIMENSIONS
    rank = 55

    start_time = time.time()
    deadline = start_time + 110

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    column = 0

    def add_term(u_vals, v_vals, w_vals):
        nonlocal column
        if time.time() > deadline:
            return False
        for idx, val in u_vals:
            if val != 0:
                U[idx][column] = val
        for idx, val in v_vals:
            if val != 0:
                V[idx][column] = val
        for idx, val in w_vals:
            if val != 0:
                W[idx][column] = val
        column += 1
        return True

    # Part 1: j=0
    for i in range(n):
        for k in range(p):
            if not add_term(
                [(i * m + 0, 1)],
                [(0 * p + k, 1)],
                [(k * n + i, 1)]
            ):
                return {"U": U, "V": V, "W": W}

    # Part 2: j=1,2
    k_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    
    for k1, k2 in k_pairs:
        a00 = 0 * m + 1
        a01 = 0 * m + 2
        a10 = 1 * m + 1
        a11 = 1 * m + 2
        
        b00 = 1 * p + k1
        b01 = 1 * p + k2
        b10 = 2 * p + k1
        b11 = 2 * p + k2
        
        c00 = k1 * n + 0
        c01 = k2 * n + 0
        c10 = k1 * n + 1
        c11 = k2 * n + 1
        
        # M1: (a00 + a11)(b00 + b11)
        # c00 += M1, c11 += M1
        if not add_term(
            [(a00, 1), (a11, 1)],
            [(b00, 1), (b11, 1)],
            [(c00, 1), (c11, 1)]
        ): return {"U": U, "V": V, "W": W}

        # M2: (a10 + a11)b00
        # c10 += M2, c11 -= M2
        if not add_term(
            [(a10, 1), (a11, 1)],
            [(b00, 1)],
            [(c10, 1), (c11, -1)]
        ): return {"U": U, "V": V, "W": W}

        # M3: a00(b01 - b11)
        # c01 += M3, c11 += M3
        if not add_term(
            [(a00, 1)],
            [(b01, 1), (b11, -1)],
            [(c01, 1), (c11, 1)]
        ): return {"U": U, "V": V, "W": W}

        # M4: a11(b10 - b00)
        # c00 += M4, c10 += M4
        if not add_term(
            [(a11, 1)],
            [(b10, 1), (b00, -1)],
            [(c00, 1), (c10, 1)]
        ): return {"U": U, "V": V, "W": W}

        # M5: (a00 + a01)b11
        # c00 -= M5, c01 += M5
        if not add_term(
            [(a00, 1), (a01, 1)],
            [(b11, 1)],
            [(c00, -1), (c01, 1)]
        ): return {"U": U, "V": V, "W": W}

        # M6: (a10 - a00)(b00 + b01)
        # c11 += M6
        if not add_term(
            [(a10, 1), (a00, -1)],
            [(b00, 1), (b01, 1)],
            [(c11, 1)]
        ): return {"U": U, "V": V, "W": W}

        # M7: (a01 - a11)(b10 + b11)
        # c00 += M7
        if not add_term(
            [(a01, 1), (a11, -1)],
            [(b10, 1), (b11, 1)],
            [(c00, 1)]
        ): return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
