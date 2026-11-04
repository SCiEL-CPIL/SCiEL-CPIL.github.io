"""Matrix multiplication 4x4 * 4x6 tensor decomposition.

Strategy: Use recursive 2x2 Strassen tiling for the 4x4*4x4 part (rank 28)
and schoolbook for the 4x4*4x2 part (rank 16). Total rank: 44.

The multiplication tensor T(A,B) = A*B where A is 4x4, B is 4x6, C is 4x6.
Decomposition: T = sum_r u_r * v_r * w_r
  u_r in R^{16} (A-space, index i*m+j)
  v_r in R^{24} (B-space, index j*p+k)
  w_r in R^{24} (C-space, index k*n+i per trace convention W[k*n+i, r])
"""

import time

DIMENSIONS = [4, 4, 6]


def construct():
    n, m, p = 4, 4, 6

    U_rows = n * m   # 16
    V_rows = m * p   # 24
    W_rows = p * n   # 24

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    def zeros(length):
        return [0] * length

    def strassen_2x2_2x2(ar, ac, br, bc, cr, cc):
        """Add 7 rank-1 terms for Strassen 2x2*2x2 at given offsets."""
        def a_idx(i, j): return (ar + i) * m + (ac + j)
        def b_idx(i, j): return (br + i) * p + (bc + j)
        def c_idx(i, j): return (cc + j) * n + (cr + i)

        # M1 = (a00 + a11)(b00 + b11) -> c00 += M1, c11 += M1
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0, 0)] = 1; u[a_idx(1, 1)] = 1
        v[b_idx(0, 0)] = 1; v[b_idx(1, 1)] = 1
        w[c_idx(0, 0)] = 1; w[c_idx(1, 1)] = 1
        add_term(u, v, w)

        # M2 = (a10 + a11)b00 -> c10 += M2, c11 -= M2
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(1, 0)] = 1; u[a_idx(1, 1)] = 1
        v[b_idx(0, 0)] = 1
        w[c_idx(1, 0)] = 1; w[c_idx(1, 1)] = -1
        add_term(u, v, w)

        # M3 = a00(b01 - b11) -> c01 += M3, c11 += M3
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0, 0)] = 1
        v[b_idx(0, 1)] = 1; v[b_idx(1, 1)] = -1
        w[c_idx(0, 1)] = 1; w[c_idx(1, 1)] = 1
        add_term(u, v, w)

        # M4 = a11(b10 - b00) -> c00 += M4, c10 += M4
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(1, 1)] = 1
        v[b_idx(1, 0)] = 1; v[b_idx(0, 0)] = -1
        w[c_idx(0, 0)] = 1; w[c_idx(1, 0)] = 1
        add_term(u, v, w)

        # M5 = (a00 + a01)b11 -> c00 -= M5, c01 += M5
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0, 0)] = 1; u[a_idx(0, 1)] = 1
        v[b_idx(1, 1)] = 1
        w[c_idx(0, 0)] = -1; w[c_idx(0, 1)] = 1
        add_term(u, v, w)

        # M6 = (a10 - a00)(b00 + b01) -> c11 += M6
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(1, 0)] = 1; u[a_idx(0, 0)] = -1
        v[b_idx(0, 0)] = 1; v[b_idx(0, 1)] = 1
        w[c_idx(1, 1)] = 1
        add_term(u, v, w)

        # M7 = (a01 - a11)(b10 + b11) -> c00 += M7
        u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
        u[a_idx(0, 1)] = 1; u[a_idx(1, 1)] = -1
        v[b_idx(1, 0)] = 1; v[b_idx(1, 1)] = 1
        w[c_idx(0, 0)] = 1
        add_term(u, v, w)

    def schoolbook_2x2_2x1(ar, ac, br, bc, cr, cc):
        """Add 4 rank-1 terms for schoolbook 2x2*2x1 at given offsets."""
        def a_idx(i, j): return (ar + i) * m + (ac + j)
        def b_idx(i, j): return (br + i) * p + (bc + j)
        def c_idx(i, j): return (cc + j) * n + (cr + i)

        for ai in range(2):
            for aj in range(2):
                u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
                u[a_idx(ai, aj)] = 1
                v[b_idx(aj, 0)] = 1
                w[c_idx(ai, 0)] = 1
                add_term(u, v, w)

    def product_2x2_2x3(ar, ac, br, bc, cr, cc):
        """Compute (2x2) A-block * (2x3) B-block = (2x3) C-block.
        Uses Strassen for first 2 columns, schoolbook for 3rd column."""
        strassen_2x2_2x2(ar, ac, br, bc, cr, cc)
        schoolbook_2x2_2x1(ar, ac, br, bc + 2, cr, cc + 2)

    # C is 4x6, A is 4x4, B is 4x6
    # Split A into 2x2 blocks: A11, A12, A21, A22
    # Split B into 2x3 blocks: B11, B12, B21, B22 (each 2x3)
    # C = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
    #      [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    # where each block is 2x3

    # C11 block: A11*B11 + A12*B21
    # A11 at (0,0), B11 at (0,0) -> C11 at (0,0)
    # A12 at (0,2), B21 at (2,0) -> C11 at (0,0)
    product_2x2_2x3(0, 0, 0, 0, 0, 0)
    product_2x2_2x3(0, 2, 2, 0, 0, 0)

    # C12 block: A11*B12 + A12*B22
    # A11 at (0,0), B12 at (0,2) -> C12 at (0,2)
    # A12 at (0,2), B22 at (2,2) -> C12 at (0,2)
    product_2x2_2x3(0, 0, 0, 2, 0, 2)
    product_2x2_2x3(0, 2, 2, 2, 0, 2)

    # C21 block: A21*B11 + A22*B21
    # A21 at (2,0), B11 at (0,0) -> C21 at (2,0)
    # A22 at (2,2), B21 at (2,0) -> C21 at (2,0)
    product_2x2_2x3(2, 0, 0, 0, 2, 0)
    product_2x2_2x3(2, 2, 2, 0, 2, 0)

    # C22 block: A21*B12 + A22*B22
    # A21 at (2,0), B12 at (0,2) -> C22 at (2,2)
    # A22 at (2,2), B22 at (2,2) -> C22 at (2,2)
    product_2x2_2x3(2, 0, 0, 2, 2, 2)
    product_2x2_2x3(2, 2, 2, 2, 2, 2)

    rank = len(U_cols)

    U = [[U_cols[r][i] for r in range(rank)] for i in range(U_rows)]
    V = [[V_cols[r][i] for r in range(rank)] for i in range(V_rows)]
    W = [[W_cols[r][i] for r in range(rank)] for i in range(W_rows)]

    return {"U": U, "V": V, "W": W}
