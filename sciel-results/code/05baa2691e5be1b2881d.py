import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    n, m, p = DIMENSIONS
    try:
        U, V, W = _strassen_4x4(n, m, p)
        return {"U": U, "V": V, "W": W}
    except Exception:
        pass

    # Fallback: Schoolbook decomposition (Rank 64)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _strassen_4x4(n, m, p):
    """
    Recursive Strassen: 4x4 = 2x2 blocks, each 2x2 block mult uses 7 products.
    Total rank = 7 (outer Strassen) * 7 (inner Strassen) = 49
    """
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Inner Strassen 2x2 decomposition (rank 7)
    # For multiplying 2x2 X * Y = Z:
    # X rows: 0,1 ; cols: 0,1
    # Y rows: 0,1 ; cols: 0,1
    # Z rows: 0,1 ; cols: 0,1
    
    # M0: (X00+X11)(Y00+Y11) -> Z00 += 1, Z11 += 1
    # M1: (X10+X11)Y00       -> Z10 += 1, Z11 -= 1
    # M2: X00(Y01-Y11)       -> Z01 += 1, Z11 += 1
    # M3: X11(Y10-Y00)       -> Z00 += 1, Z10 += 1
    # M4: (X00+X01)Y11       -> Z01 += 1, Z00 -= 1
    # M5: (X10-X00)(Y00+Y01) -> Z11 += 1
    # M6: (X01-X11)(Y10+Y11) -> Z00 += 1

    inner_defs = [
        {'u': [(0,0,1),(1,1,1)], 'v': [(0,0,1),(1,1,1)], 'w': [(0,0,1),(1,1,1)]},
        {'u': [(1,0,1),(1,1,1)], 'v': [(0,0,1)], 'w': [(1,0,1),(1,1,-1)]},
        {'u': [(0,0,1)], 'v': [(0,1,1),(1,1,-1)], 'w': [(0,1,1),(1,1,1)]},
        {'u': [(1,1,1)], 'v': [(1,0,1),(0,0,-1)], 'w': [(0,0,1),(1,0,1)]},
        {'u': [(0,0,1),(0,1,1)], 'v': [(1,1,1)], 'w': [(0,1,1),(0,0,-1)]},
        {'u': [(1,0,1),(0,0,-1)], 'v': [(0,0,1),(0,1,1)], 'w': [(1,1,1)]},
        {'u': [(0,1,1),(1,1,-1)], 'v': [(1,0,1),(1,1,1)], 'w': [(0,0,1)]},
    ]

    # Outer Strassen for 4x4 (treating as 2x2 blocks of size 2x2)
    outer_defs = [
        {'a_blocks': [(0,0,1),(1,1,1)], 'b_blocks': [(0,0,1),(1,1,1)], 'c_blocks': [(0,0,1),(1,1,1)]},
        {'a_blocks': [(1,0,1),(1,1,1)], 'b_blocks': [(0,0,1)], 'c_blocks': [(1,0,1),(1,1,-1)]},
        {'a_blocks': [(0,0,1)], 'b_blocks': [(0,1,1),(1,1,-1)], 'c_blocks': [(0,1,1),(1,1,1)]},
        {'a_blocks': [(1,1,1)], 'b_blocks': [(1,0,1),(0,0,-1)], 'c_blocks': [(0,0,1),(1,0,1)]},
        {'a_blocks': [(0,0,1),(0,1,1)], 'b_blocks': [(1,1,1)], 'c_blocks': [(0,1,1),(0,0,-1)]},
        {'a_blocks': [(1,0,1),(0,0,-1)], 'b_blocks': [(0,0,1),(0,1,1)], 'c_blocks': [(1,1,1)]},
        {'a_blocks': [(0,1,1),(1,1,-1)], 'b_blocks': [(1,0,1),(1,1,1)], 'c_blocks': [(0,0,1)]},
    ]

    for s, outer in enumerate(outer_defs):
        for t, inner in enumerate(inner_defs):
            r = s * 7 + t

            # Build U contributions
            for (P, Q, sign_a) in outer['a_blocks']:
                for (ir, jc, sign_x) in inner['u']:
                    coeff = sign_a * sign_x
                    if coeff != 0:
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        u_idx = row * m + col
                        U[u_idx][r] += coeff

            # Build V contributions
            for (P, Q, sign_b) in outer['b_blocks']:
                for (ir, jc, sign_y) in inner['v']:
                    coeff = sign_b * sign_y
                    if coeff != 0:
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        v_idx = row * p + col
                        V[v_idx][r] += coeff

            # Build W contributions
            for (P, Q, sign_c) in outer['c_blocks']:
                for (ir, jc, sign_z) in inner['w']:
                    coeff = sign_c * sign_z
                    if coeff != 0:
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        w_idx = col * n + row
                        W[w_idx][r] += coeff

    return U, V, W
