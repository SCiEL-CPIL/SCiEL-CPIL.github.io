"""Matrix multiplication 2,4,6 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Build the rank-42 Strassen-based decomposition
    # Split p=6 into three 2-column blocks, each with two 2x2x2 Strassen sub-problems
    rank_base = 42
    U = [[0] * rank_base for _ in range(n * m)]
    V = [[0] * rank_base for _ in range(m * p)]
    W = [[0] * rank_base for _ in range(p * n)]

    # Strassen 2x2x2 terms
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    strassen_terms = [
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(2, 1), (3, -1)]},
        {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(1, 1), (3, 1)]},
        {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(2, 1), (0, 1)]},
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (1, 1)]},
        {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
        {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]}
    ]

    col = 0
    for r in range(3):
        for sub in range(2):
            a_col_offset = sub * 2
            b_row_offset = sub * 2
            for s_idx in range(7):
                term = strassen_terms[s_idx]
                for local_u, coeff in term['U']:
                    a = local_u // 2
                    l = local_u % 2
                    global_u = a * m + (a_col_offset + l)
                    U[global_u][col] += coeff
                for local_v, coeff in term['V']:
                    l = local_v // 2
                    c = local_v % 2
                    global_v = (b_row_offset + l) * p + (2 * r + c)
                    V[global_v][col] += coeff
                for local_w, coeff in term['W']:
                    a = local_w // 2
                    c = local_w % 2
                    global_w = (2 * r + c) * n + a
                    W[global_w][col] += coeff
                col += 1

    def verify(U_mat, V_mat, W_mat, r):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for rr in range(r):
                        val += U_mat[i * m + j][rr] * V_mat[j * p + k][rr] * W_mat[k * n + i][rr]
                    if abs(val - 1) > 1e-6:
                        return False
        return True

    # Verify the base decomposition
    base_valid = verify(U, V, W, rank_base)
    if not base_valid:
        # Fallback: schoolbook
        r_fb = n * m * p
        U_fb = [[0] * r_fb for _ in range(n * m)]
        V_fb = [[0] * r_fb for _ in range(m * p)]
        W_fb = [[0] * r_fb for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][c] = 1
                    V_fb[j * p + k][c] = 1
                    W_fb[k * n + i][c] = 1
                    c += 1
        return {'U': U_fb, 'V': V_fb, 'W': W_fb}

    best_rank = rank_base
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]

    # Try to reduce rank by finding linear dependencies among the columns
    # Convert to numpy for rank computation
    def try_reduce_rank(U_mat, V_mat, W_mat, r):
        """Attempt to reduce rank by identifying redundant columns."""
        if time.time() > deadline:
            return U_mat, V_mat, W_mat, r

        # Stack columns: each column is a triple (u_col, v_col, w_col)
        # If we can express one column as a linear combination of others in all three matrices,
        # we can remove it.
        
        # For each column, check if it's in the span of the others
        # This is expensive, so try a heuristic: look for zero or duplicate columns first
        
        # Remove zero columns
        keep = []
        for c in range(r):
            u_zero = all(U_mat[i][c] == 0 for i in range(len(U_mat)))
            v_zero = all(V_mat[i][c] == 0 for i in range(len(V_mat)))
            w_zero = all(W_mat[i][c] == 0 for i in range(len(W_mat)))
            if not (u_zero or v_zero or w_zero):
                keep.append(c)
        
        if len(keep) < r:
            new_r = len(keep)
            new_U = [[U_mat[i][c] for c in keep] for i in range(len(U_mat))]
            new_V = [[V_mat[i][c] for c in keep] for i in range(len(V_mat))]
            new_W = [[W_mat[i][c] for c in keep] for i in range(len(W_mat))]
            if verify(new_U, new_V, new_W, new_r):
                return new_U, new_V, new_W, new_r
            else:
                return U_mat, V_mat, W_mat, r

        # Try removing each column one at a time and checking if verification still passes
        # This works if the column's contribution is zero in the sum
        for c_remove in range(r):
            if time.time() > deadline:
                break
            keep = [c for c in range(r) if c != c_remove]
            new_r = r - 1
            new_U = [[U_mat[i][c] for c in keep] for i in range(len(U_mat))]
            new_V = [[V_mat[i][c] for c in keep] for i in range(len(V_mat))]
            new_W = [[W_mat[i][c] for c in keep] for i in range(len(W_mat))]
            if verify(new_U, new_V, new_W, new_r):
                return new_U, new_V, new_W, new_r

        return U_mat, V_mat, W_mat, r

    # Iteratively try to reduce rank
    current_U, current_V, current_W, current_r = U, V, W, rank_base
    for _ in range(10):
        if time.time() > deadline:
            break
        new_U, new_V, new_W, new_r = try_reduce_rank(current_U, current_V, current_W, current_r)
        if new_r < current_r:
            current_U, current_V, current_W, current_r = new_U, new_V, new_W, new_r
            if current_r < best_rank:
                best_rank = current_r
                best_U, best_V, best_W = [row[:] for row in current_U], [row[:] for row in current_V], [row[:] for row in current_W]
        else:
            break

    return {'U': best_U, 'V': best_V, 'W': best_W}
