import numpy as np
import time

def construct():
    n, m, p = 3, 4, 4
    deadline = time.time() + 120

    def _strassen_2x2_terms():
        """Return the 7 Strassen terms as (U_coeffs, V_coeffs, W_coeffs) for 2x2x2.
        
        For X(2x2) * Y(2x2) = Z(2x2):
        M1 = (X00+X11)(Y00+Y11)
        M2 = (X10+X11)(Y00)
        M3 = (X00)(Y01-Y11)
        M4 = (X11)(Y10-Y00)
        M5 = (X00+X01)(Y11)
        M6 = (X10-X00)(Y00+Y01)
        M7 = (X01-X11)(Y10+Y11)
        
        Z00 = M1 + M4 - M5 + M7
        Z01 = M3 + M5
        Z10 = M2 + M4
        Z11 = M1 - M2 + M3 + M6
        """
        terms = [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
        ]
        return terms

    def _add_strassen_block(U, V, W, a_row_off, a_col_off, b_row_off, b_col_off,
                            c_row_off, c_col_off, offset, m, p, n):
        """Add a 2x2 Strassen block at the given offsets."""
        terms = _strassen_2x2_terms()
        for t_idx, (x_coeff, y_coeff, z_coeff) in enumerate(terms):
            r = offset + t_idx
            for (xi, xj), val in x_coeff.items():
                U[(a_row_off + xi) * m + (a_col_off + xj)][r] += val
            for (yi, yj), val in y_coeff.items():
                V[(b_row_off + yi) * p + (b_col_off + yj)][r] += val
            for (zi, zj), val in z_coeff.items():
                W[(c_col_off + zj) * n + (c_row_off + zi)][r] += val

    def _verify(U, V, W, tol=1e-9):
        """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k"""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def _make_rank32():
        """32-rank decomposition: 28 Strassen (4 blocks of 2x2x2) + 4 schoolbook for row 2."""
        rank = 32
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        blocks = [(0, 0, 0, 0, 0, 0), (0, 0, 0, 2, 0, 2),
                  (0, 2, 2, 0, 0, 0), (0, 2, 2, 2, 0, 2)]
        base = 0
        for (ar, ac, br, bc, cr, cc) in blocks:
            _add_strassen_block(U, V, W, ar, ac, br, bc, cr, cc, base, m, p, n)
            base += 7

        base = 28
        for r in range(4):
            col_idx = base + r
            for j in range(4):
                U[2 * m + j][col_idx] = 1.0
            for j in range(4):
                for k in range(4):
                    if k == r:
                        V[j * p + k][col_idx] = 1.0
            for k in range(4):
                if k == r:
                    W[k * n + 2][col_idx] = 1.0
        return U, V, W

    def _make_rank44():
        """44-rank decomposition: 16 schoolbook for row 0 + 28 Strassen for rows 1-2."""
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0
        for j in range(4):
            for k in range(4):
                U[0 * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + 0][col] = 1
                col += 1

        blocks = [
            (1, 0, 0, 0, 1, 0),
            (1, 0, 0, 2, 1, 2),
            (1, 2, 2, 0, 1, 0),
            (1, 2, 2, 2, 1, 2),
        ]
        for (ar, ac, br, bc, cr, cc) in blocks:
            _add_strassen_block(U, V, W, ar, ac, br, bc, cr, cc, col, m, p, n)
            col += 7

        return U, V, W

    def _make_schoolbook():
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        return U_sb, V_sb, W_sb

    # Try rank 32 first (best known)
    best_result = None
    U32, V32, W32 = _make_rank32()
    if _verify(U32, V32, W32):
        best_result = (32, U32, V32, W32)

    # Try rank 44 as a fallback
    if best_result is None:
        U44, V44, W44 = _make_rank44()
        if _verify(U44, V44, W44):
            best_result = (44, U44, V44, W44)

    if best_result is not None:
        rank, U, V, W = best_result
        return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook
    U_sb, V_sb, W_sb = _make_schoolbook()
    return {"U": U_sb, "V": V_sb, "W": W_sb}
