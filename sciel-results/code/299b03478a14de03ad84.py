"""Matrix multiplication 4x4 * 4x6 tensor decomposition.

Strategy: Recursive Strassen for the 4x4*4x4 part (rank 49) plus
block Strassen for the 4x4*4x2 part (rank 28), total rank 77.

The 4x4*4x4 uses top-level Strassen (7 terms), each producing a 2x2*2x2
block product computed via Strassen (7 terms), giving 7*7 = 49.
The 4x4*4x2 splits into four 2x2*2x2 block products, each using
Strassen (7 terms), giving 4*7 = 28.

Total rank: 49 + 28 = 77.
"""

import time
import numpy as np


def _build_rank7_factors():
    """Return (U, V, W) for 2x2*2x2 Strassen decomposition, each shape (4,7).
    Indexing: U[i*2+j], V[j*2+k], W[i*2+k] (W uses i=0..1 row, k=0..1 col)."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    # Strassen products: M1..M7
    # M1 = (a+d)(e+f): U has a,d in A; V has e,f in B; W has a,d in C
    # M2 = (b+d)e: U has b,d; V has e; W has b in C
    # M3 = a(f-g): U has a; V has f,-g; W has f in C
    # M4 = d(c-e): U has d; V has c,-e; W has c in C
    # M5 = (a+b)(g+h): U has a,b; V has g,h; W has a,-b in C (a+h - (a+b)? no)
    #   Actually M5 contributes to c[1,1] = a+b - c - d... let me use standard:
    #   c[0,0] = M1 - M4 + M5 - M7
    #   c[0,1] = M3 + M7
    #   c[1,0] = M2 + M4
    #   c[1,1] = M1 - M3 + M5 - M6
    # where:
    #   M1 = (a+d)(e+f)
    #   M2 = (b+d)e
    #   M3 = a(f-g)
    #   M4 = d(c-e)
    #   M5 = (a+b)(g+h)
    #   M6 = (c-b)(e+f)
    #   M7 = (a-c)(d+e)

    # Index convention: A[i,j], B[j,k], C[i,k] with i,j,k in {0,1}
    # U[i*2+j], V[j*2+k], W[i*2+k]

    prods = [
        # M1: (a+d)(e+f)
        ([(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (0,1,1.0)], [(0,0,1.0), (1,1,1.0)]),
        # M2: (b+d)e
        ([(0,1,1.0), (1,1,1.0)], [(0,0,1.0)], [(0,1,1.0), (1,1,-1.0)]),
        # M3: a(f-g)
        ([(0,0,1.0)], [(0,1,1.0), (1,1,-1.0)], [(0,1,1.0), (1,1,1.0)]),
        # M4: d(c-e)
        ([(1,1,1.0)], [(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (1,0,1.0)]),
        # M5: (a+b)(g+h)
        ([(0,0,1.0), (0,1,1.0)], [(1,1,1.0)], [(0,0,-1.0), (0,1,1.0)]),
        # M6: (c-b)(e+f)
        ([(1,0,1.0), (0,1,-1.0)], [(0,0,1.0), (0,1,1.0)], [(1,1,1.0)]),
        # M7: (a-c)(d+e)
        ([(0,0,1.0), (1,1,-1.0)], [(1,0,1.0), (1,1,1.0)], [(0,0,1.0)]),
    ]

    for m_idx, (Ub, Vb, Wb) in enumerate(prods):
        for (i, j, c) in Ub:
            U[i * 2 + j, m_idx] += c
        for (j, k, c) in Vb:
            V[j * 2 + k, m_idx] += c
        for (i, k, c) in Wb:
            W[i * 2 + k, m_idx] += c

    return U, V, W


def construct():
    n, m, p = 4, 4, 6
    t0 = time.time()
    deadline = t0 + 110.0

    U_s, V_s, W_s = _build_rank7_factors()

    rank_444 = 49  # 7 outer * 7 inner
    rank_442 = 28  # 4 blocks * 7
    total_rank = rank_444 + rank_442

    U = np.zeros((n * m, total_rank))  # 16 x 77
    V = np.zeros((m * p, total_rank))  # 24 x 77
    W = np.zeros((p * n, total_rank))  # 24 x 77

    # Part 1: 4x4 * 4x4 using recursive Strassen (7x7 = 49 terms)
    # Outer Strassen splits 4x4 into 2x2 blocks (I,J) x (J,K)
    # Inner Strassen computes each 2x2*2x2 product
    for t in range(7):  # outer Strassen term
        for s in range(7):  # inner Strassen term
            col = 7 * t + s
            # U part: outer term t selects which 2x2 subblock of A
            # Inner term s selects which entry within that 2x2 block
            for I in range(2):
                for J in range(2):
                    outer_u = U_s[2 * I + J, t]
                    if outer_u == 0.0:
                        continue
                    for i0 in range(2):
                        for j0 in range(2):
                            inner_u = U_s[2 * i0 + j0, s]
                            if inner_u == 0.0:
                                continue
                            val = outer_u * inner_u
                            i = 2 * I + i0
                            j = 2 * J + j0
                            U[i * 4 + j, col] += val

            # V part
            for J in range(2):
                for K in range(2):
                    outer_v = V_s[2 * J + K, t]
                    if outer_v == 0.0:
                        continue
                    for j0 in range(2):
                        for k0 in range(2):
                            inner_v = V_s[2 * j0 + k0, s]
                            if inner_v == 0.0:
                                continue
                            val = outer_v * inner_v
                            j = 2 * J + j0
                            k = 2 * K + k0
                            V[j * 6 + k, col] += val

            # W part: output C[i,k], index k*4 + i (trace convention W[k*n+i])
            for K in range(2):
                for I in range(2):
                    outer_w = W_s[2 * I + K, t]
                    if outer_w == 0.0:
                        continue
                    for i0 in range(2):
                        for k0 in range(2):
                            inner_w = W_s[2 * i0 + k0, s]
                            if inner_w == 0.0:
                                continue
                            val = outer_w * inner_w
                            k = 2 * K + k0
                            i = 2 * I + i0
                            W[k * 4 + i, col] += val

    # Part 2: 4x4 * 4x2 for columns 4,5 of B
    # Four 2x2*2x2 block products, each using Strassen (rank 7)
    block_products = [
        # (A_row_off, A_col_off, B_row_off, B_col_off, C_row_off, C_col_off)
        (0, 0, 0, 4, 0, 4),   # A[0:2,0:2] * B[0:2,4:6] -> C[0:2,4:6]
        (0, 2, 2, 4, 0, 4),   # A[0:2,2:4] * B[2:4,4:6] -> C[0:2,4:6]
        (2, 0, 0, 4, 2, 4),   # A[2:4,0:2] * B[0:2,4:6] -> C[2:4,4:6]
        (2, 2, 2, 4, 2, 4),   # A[2:4,2:4] * B[2:4,4:6] -> C[2:4,4:6]
    ]

    for block_idx, (ar, ac, br, bc, cr, cc) in enumerate(block_products):
        for s in range(7):
            col = rank_444 + block_idx * 7 + s
            # U part: A[ar+i, ac+j]
            for i in range(2):
                for j in range(2):
                    val = U_s[i * 2 + j, s]
                    if val != 0.0:
                        U[(ar + i) * 4 + (ac + j), col] += val

            # V part: B[br+j, bc+k]
            for j in range(2):
                for k in range(2):
                    val = V_s[j * 2 + k, s]
                    if val != 0.0:
                        V[(br + j) * 6 + (bc + k), col] += val

            # W part: C[cr+i, cc+k], index (cc+k)*4 + (cr+i)
            for i in range(2):
                for k in range(2):
                    val = W_s[i * 2 + k, s]
                    if val != 0.0:
                        W[(cc + k) * 4 + (cr + i), col] += val

    # Check deadline
    if time.time() > deadline:
        raise TimeoutError("Exceeded deadline")

    # Convert to list with clean integer values
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for x in row:
                r = round(x)
                if abs(x - r) < 1e-9:
                    new_row.append(int(r))
                else:
                    new_row.append(float(x))
            result.append(new_row)
        return result

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
