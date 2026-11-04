"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe (150s wall-clock)

    n, m, p = DIMENSIONS
    N_M = n * m  # 15
    M_P = m * p  # 35
    P_N = p * n  # 21

    # Helper to add a Strassen 2x2x2 block
    def add_strassen(U, V, W, i0, j0, k0, col):
        # M1
        U[i0 * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1

        # M2
        U[(i0 + 1) * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] -= 1
        col += 1

        # M3
        U[i0 * m + j0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] -= 1
        W[(k0 + 1) * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1

        # M4
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0][col] += 1
        V[j0 * p + k0][col] -= 1
        W[k0 * n + i0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        col += 1

        # M5
        U[i0 * m + j0][col] += 1
        U[i0 * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0][col] += 1
        W[k0 * n + i0][col] -= 1
        col += 1

        # M6
        U[(i0 + 1) * m + j0][col] += 1
        U[i0 * m + j0][col] -= 1
        V[j0 * p + k0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1

        # M7
        U[i0 * m + j0 + 1][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] -= 1
        V[(j0 + 1) * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        col += 1

        return col

    # Helper to add schoolbook terms for a sub-block
    def add_schoolbook(U, V, W, i_range, j_range, k_range, col):
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return col

    # Build the baseline Strassen+Schoolbook decomposition (rank 99)
    rank_baseline = 99
    U_base = [[0] * rank_baseline for _ in range(N_M)]
    V_base = [[0] * rank_baseline for _ in range(M_P)]
    W_base = [[0] * rank_baseline for _ in range(P_N)]

    col = 0
    # Part 1: j in {0, 1}
    for k0 in [0, 2, 4]:
        col = add_strassen(U_base, V_base, W_base, 0, 0, k0, col)
    col = add_schoolbook(U_base, V_base, W_base, [0, 1], [0, 1], [6], col)
    col = add_schoolbook(U_base, V_base, W_base, [2], [0, 1], range(7), col)

    # Part 2: j in {2, 3, 4}
    for k0 in [0, 2, 4]:
        col = add_strassen(U_base, V_base, W_base, 0, 2, k0, col)
    col = add_schoolbook(U_base, V_base, W_base, [0, 1], [2, 3], [6], col)
    col = add_schoolbook(U_base, V_base, W_base, [0, 1], [4], range(7), col)
    col = add_schoolbook(U_base, V_base, W_base, [2], [2, 3, 4], range(7), col)

    # Build the constraint matrix for the baseline solution
    def build_constraint_matrix(U, V, W, n, m, p):
        """Build the 105 x rank matrix where entry (i,j,k),r = U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]"""
        r = len(U[0])
        M = [[Fraction(0)] * r for _ in range(n * m * p)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r_idx in range(r):
                        M[idx][r_idx] = Fraction(U[i * m + j][r_idx]) * Fraction(V[j * p + k][r_idx]) * Fraction(W[k * n + i][r_idx])
                    idx += 1
        return M

    def find_redundant_columns_with_coefficients(M):
        """
        Perform Gaussian elimination and track which columns are redundant.
        For each redundant column, compute its expression as a linear combination of pivot columns.
        Returns (pivot_cols, redundant_info) where redundant_info maps col_idx -> {pivot_idx: coeff}
        """
        if not M or not M[0]:
            return [], {}
        rows = len(M)
        cols = len(M[0])
        A = [row[:] for row in M]
        pivot_cols = []
        redundant_info = {}

        current_row = 0
        for col_idx in range(cols):
            if time.time() > deadline:
                break
            if current_row >= rows:
                # This column is redundant; express as combination of previous pivots
                coeff_dict = {}
                for r_idx in range(current_row):
                    if A[r_idx][col_idx] != 0:
                        coeff_dict[pivot_cols[r_idx]] = A[r_idx][col_idx]
                redundant_info[col_idx] = coeff_dict
                continue
            # Find pivot in column col_idx, row >= current_row
            pivot_row = -1
            for row_idx in range(current_row, rows):
                if A[row_idx][col_idx] != 0:
                    pivot_row = row_idx
                    break
            if pivot_row == -1:
                # This column is redundant
                coeff_dict = {}
                for r_idx in range(current_row):
                    if A[r_idx][col_idx] != 0:
                        coeff_dict[pivot_cols[r_idx]] = A[r_idx][col_idx]
                redundant_info[col_idx] = coeff_dict
                continue
            # Swap
            A[current_row], A[pivot_row] = A[pivot_row], A[current_row]
            # Eliminate below
            pivot_val = A[current_row][col_idx]
            for row_idx in range(current_row + 1, rows):
                if A[row_idx][col_idx] != 0:
                    factor = A[row_idx][col_idx] / pivot_val
                    for c in range(col_idx, cols):
                        A[row_idx][c] -= factor * A[current_row][c]
            pivot_cols.append(col_idx)
            current_row += 1

        return pivot_cols, redundant_info

    # Build the constraint matrix for the baseline solution
    M_base = build_constraint_matrix(U_base, V_base, W_base, n, m, p)

    # Find redundant columns
    pivot_cols, redundant_info = find_redundant_columns_with_coefficients(M_base)

    # Now we need to find a subset of columns that sums to the all-ones vector.
    # The original sum: sum_r T^r = T_target (all ones)
    # If T^r = sum_{s in pivot} c_{r,s} T^s, then:
    # sum_r T^r = sum_{s in pivot} T^s + sum_{r redundant} sum_{s in pivot} c_{r,s} T^s
    #           = sum_{s in pivot} (1 + sum_{r redundant} c_{r,s}) T^s
    # We want this to equal T_target.
    # But actually, we want to find coefficients d_s such that sum_s d_s T^s = T_target
    # and use as few terms as possible.

    # A simpler approach: try to eliminate redundant columns one by one.
    # For each redundant column r with T^r = sum_{s in pivot} c_{r,s} T^s:
    # The contribution of T^r to the sum is T^r.
    # If we remove T^r and add sum_{s in pivot} c_{r,s} T^s instead, the total sum remains the same.
    # But this doesn't reduce the number of terms unless we can merge.

    # Actually, the key insight: if we have a linear dependency sum_r d_r T^r = 0,
    # then we can modify the coefficients. Originally, each term has coefficient 1.
    # If sum_r d_r T^r = 0, then sum_r (1 + d_r) T^r = sum_r T^r = T_target.
    # So we can change the coefficient of any term r to 1 + d_r.
    # If for some r, 1 + d_r = 0, we can eliminate that term entirely.

    # Let's find such dependencies.
    # From the Gaussian elimination, if column r is redundant with T^r = sum_{s in pivot} c_{r,s} T^s,
    # then T^r - sum_{s in pivot} c_{r,s} T^s = 0.
    # This gives d_r = 1, d_s = -c_{r,s} for s in pivot.
    # Then 1 + d_r = 2 (not zero), and 1 + d_s = 1 - c_{r,s} (might be zero for some s).

    # Let's try a different approach: directly solve for a reduced decomposition.
    # We want to find the minimum number of terms r such that there exist
    # U (15 x r), V (35 x r), W (21 x r) with sum_{r'} U[i*m+j][r'] * V[j*p+k][r'] * W[k*n+i][r'] = 1

    # This is hard. Let's try a greedy approach:
    # Start with all 99 terms. For each term, check if it can be removed while maintaining
    # the sum = all-ones. This requires solving a linear system.

    # Simpler: try specific structured improvements.
    # The 2x2x7 block uses 25 terms (3*7 + 4). Can we do better?
    # Known: 2x2x3 might be doable in 11 terms instead of 13.

    # For now, let's just return the baseline if we can't find improvements quickly.
    # But first, let's try one more thing: check if any of the 99 terms can be eliminated
    # by finding a dependency where 1 + d_r = 0 for some r.

    # Actually, let's just return the baseline for now. The linear algebra approach
    # is complex and may not yield improvements within the time budget.
    return {"U": U_base, "V": V_base, "W": W_base}
# EVOLVE-BLOCK-END
