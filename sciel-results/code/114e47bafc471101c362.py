"""
Winograd-style decomposition for 2x4 * 4x5 matrix multiplication.

Approach:
  Split A (2x4) into A_L (2x2, cols 0-1) and A_R (2x2, cols 2-3).
  Split B (4x5) into B_T (2x5, rows 0-1) and B_B (2x5, rows 2-3).
  C = A_L * B_T + A_R * B_B.

  For each 2x2 * 2x5 block, we use a Winograd/Fourier-based decomposition:
  - Split the 2x5 B-block into B_01 (2x2), B_23 (2x2), B_4 (2x1).
  - Use Strassen (rank 7) for the 2x2*2x2 parts.
  - Use schoolbook (rank 4) for the 2x2*2x1 part.
  - Total per 2x2*2x5: 7 + 7 + 4 = 18.
  - Total: 2 * 18 = 36.

  This is the same rank as the seed program but with a cleaner structure
  that could be further optimized if better sub-tensor decompositions exist.
"""

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    col = [0]  # use list for mutable nonlocal

    def add_rank1(u_dict, v_dict, w_dict):
        if col[0] >= rank:
            raise ValueError("Rank exceeded")
        c = col[0]
        for idx, val in u_dict.items():
            U[idx][c] = val
        for idx, val in v_dict.items():
            V[idx][c] = val
        for idx, val in w_dict.items():
            W[idx][c] = val
        col[0] += 1

    # Strassen for 2x2 * 2x2
    def strassen(a_i0, a_j0, b_i0, b_j0, c_i0, c_k0):
        # A block at rows a_i0..a_i0+1, cols a_j0..a_j0+1
        # B block at rows b_i0..b_i0+1, cols b_j0..b_j0+1
        # C block at rows c_i0..c_i0+1, cols c_k0..c_k0+1
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i

        def ua(li):
            return (a_i0 + li // 2) * m + (a_j0 + li % 2)

        def vb(li):
            return (b_i0 + li // 2) * p + (b_j0 + li % 2)

        def wc(li):
            i = c_i0 + li // 2
            k = c_k0 + li % 2
            return k * n + i

        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        add_rank1({ua(0): 1, ua(3): 1}, {vb(0): 1, vb(3): 1}, {wc(0): 1, wc(3): 1})
        # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        add_rank1({ua(2): 1, ua(3): 1}, {vb(0): 1}, {wc(2): 1, wc(3): -1})
        # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
        add_rank1({ua(0): 1}, {vb(1): 1, vb(3): -1}, {wc(1): 1, wc(3): 1})
        # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
        add_rank1({ua(3): 1}, {vb(2): 1, vb(0): -1}, {wc(0): 1, wc(2): 1})
        # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
        add_rank1({ua(0): 1, ua(1): 1}, {vb(3): 1}, {wc(0): -1, wc(1): 1})
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        add_rank1({ua(2): 1, ua(0): -1}, {vb(0): 1, vb(1): 1}, {wc(3): 1})
        # M7 = (A01-A11)(B10+B11) -> C00 += M7
        add_rank1({ua(1): 1, ua(3): -1}, {vb(2): 1, vb(3): 1}, {wc(0): 1})

    # Schoolbook for 2x2 * 2x1
    def schoolbook_2x2_1(a_i0, a_j0, b_i0, b_j0, c_i0, c_k0):
        # A: 2x2 at (a_i0, a_j0), B: 2x1 at (b_i0, b_j0), C: 2x1 at (c_i0, c_k0)
        # C_i = A_i0*B_0 + A_i1*B_1 for i=0,1
        # 4 rank-1 terms

        def ua(li):
            return (a_i0 + li // 2) * m + (a_j0 + li % 2)

        def vb(li):
            return (b_i0 + li) * p + b_j0

        def wc(li):
            i = c_i0 + li
            k = c_k0
            return k * n + i

        # C_0 = A_00*B_0 + A_01*B_1
        add_rank1({ua(0): 1}, {vb(0): 1}, {wc(0): 1})
        add_rank1({ua(1): 1}, {vb(1): 1}, {wc(0): 1})
        # C_1 = A_10*B_0 + A_11*B_1
        add_rank1({ua(2): 1}, {vb(0): 1}, {wc(1): 1})
        add_rank1({ua(3): 1}, {vb(1): 1}, {wc(1): 1})

    # Block 1: A_L (rows 0-1, cols 0-1) * B_T (rows 0-1, cols 0-4)
    # B_T split: cols 0-1, cols 2-3, col 4
    strassen(0, 0, 0, 0, 0, 0)   # A_L * B_T[0:2, 0:2]
    strassen(0, 0, 0, 2, 0, 2)   # A_L * B_T[0:2, 2:4]
    schoolbook_2x2_1(0, 0, 0, 4, 0, 4)  # A_L * B_T[0:2, 4]

    # Block 2: A_R (rows 0-1, cols 2-3) * B_B (rows 2-3, cols 0-4)
    # B_B split: cols 0-1, cols 2-3, col 4
    strassen(0, 2, 2, 0, 0, 0)   # A_R * B_B[2:4, 0:2]
    strassen(0, 2, 2, 2, 0, 2)   # A_R * B_B[2:4, 2:4]
    schoolbook_2x2_1(0, 2, 2, 4, 0, 4)  # A_R * B_B[2:4, 4]

    return {"U": U, "V": V, "W": W}
