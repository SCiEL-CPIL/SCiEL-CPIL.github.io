"""Block Strassen decomposition for 2x4 times 4x8 matrix multiplication.

A (2x4) and B (4x8) are split into 2x2 block matrices (A blocks 1x2, B blocks 2x4).
Strassen's 7-product algorithm is applied at the block level; each block product
(1x2)x(2x4) uses the rank-8 schoolbook decomposition. Linearity lets U and V be
single (linear-combination) vectors per rank-1 term, giving rank 7*2*4 = 56.
"""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    # Strassen coefficients on A blocks (A00,A01,A10,A11) and B blocks (B00,B01,B10,B11).
    M_defs = [
        ([1, 0, 0, -1], [0, 0, 1, 1]),    # M0=(A00-A11)(B10+B11)
        ([1, 0, 0, 1],  [1, 0, 0, 0]),    # M1=(A00+A11)B00
        ([0, 0, 0, 1],  [0, 1, 0, -1]),   # M2=A11(B01-B11)
        ([1, 0, 0, 0],  [-1, 0, 0, 1]),   # M3=A00(B11-B00)
        ([1, 0, 1, 0],  [1, 0, 0, 0]),    # M4=(A00+A10)B00
        ([0, 0, 1, 1],  [0, 1, 0, -1]),   # M5=(A10+A11)(B01-B11)
        ([0, 1, 0, -1], [1, 0, 1, 0]),    # M6=(A01-A11)(B00+B10)
    ]
    # C block (i*2+c) = sum_t C_defs[i*2+c][t] * M_t
    C_defs = [
        [1, 0, 0, 1, -1, 0, 1],   # C00 = M0 + M3 - M4 + M6
        [0, 0, 1, 0, 1, 0, 0],    # C01 = M2 + M4
        [0, 1, 0, 1, 0, 0, 0],    # C10 = M1 + M3
        [1, -1, 1, 0, 0, 1, 0],   # C11 = M0 - M1 + M2 + M5
    ]

    # Rank-1 terms indexed by (t, a, cc): t in [0,7), a in [0,2), cc in [0,4).
    total_rank = 7 * 2 * 4
    U = [[0] * total_rank for _ in range(n * m)]   # 8 rows
    V = [[0] * total_rank for _ in range(m * p)]   # 32 rows
    W = [[0] * total_rank for _ in range(p * n)]   # 16 rows

    # q -> (i, j) for A block; r -> (j, c) for B block.
    q_ij = [(0, 0), (0, 1), (1, 0), (1, 1)]
    r_jc = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for t in range(7):
        alpha = M_defs[t][0]
        beta = M_defs[t][1]
        for a in range(2):
            for cc in range(4):
                col = t * 8 + a * 4 + cc
                b = a  # for (1x2)x(2x4): C_cc = sum_a A_a B_{a,cc}
                # U: linear combo over A blocks q of alpha[q] * e_{i_q, j_q, a}
                for q in range(4):
                    if alpha[q] == 0:
                        continue
                    i_q, j_q = q_ij[q]
                    U[i_q * 4 + 2 * j_q + a][col] += alpha[q]
                # V: linear combo over B blocks r of beta[r] * e_{j_r, c_r, (b, cc)}
                for r in range(4):
                    if beta[r] == 0:
                        continue
                    j_r, c_r = r_jc[r]
                    V[(2 * j_r + b) * 8 + 4 * c_r + cc][col] += beta[r]
                # W: the output C block for term (a, cc) is NOT unique in general;
                # but for (1x2)x(2x4) the output C_cc depends on which (i, c) block.
                # Since U spans all A blocks and V spans all B blocks, the product U*V
                # expands over all (q, r), and each (q, r) writes to C block (i_q, c_r).
                # We must add W contributions per (q, r) with C_defs[i_q*2+c_r][t].
                # So W cannot be a single vector; it must accumulate over (q, r).
                for q in range(4):
                    if alpha[q] == 0:
                        continue
                    i_q, _ = q_ij[q]
                    for r in range(4):
                        if beta[r] == 0:
                            continue
                        _, c_r = r_jc[r]
                        ccomb = C_defs[i_q * 2 + c_r][t]
                        if ccomb == 0:
                            continue
                        W[(4 * c_r + cc) * 2 + i_q][col] += ccomb

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
