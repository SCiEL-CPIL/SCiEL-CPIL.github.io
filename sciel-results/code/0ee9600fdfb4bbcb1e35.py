import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Partition 5 into 2, 2, 1
    parts = [2, 2, 1]
    offsets = [0, 2, 4]

    # Strassen factors for 2x2x2
    strassen_factors = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    # Calculate total rank
    total_rank = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                si, sj, sk = parts[i], parts[j], parts[k]
                if si == 2 and sj == 2 and sk == 2:
                    total_rank += 7
                else:
                    total_rank += si * sj * sk

    # Allocate matrices
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)

    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                si, sj, sk = parts[i], parts[j], parts[k]
                off_i, off_j, off_k = offsets[i], offsets[j], offsets[k]

                if si == 2 and sj == 2 and sk == 2:
                    for uf, vf, wf in strassen_factors:
                        for idx in range(4):
                            if uf[idx] != 0:
                                lr, lc = idx // 2, idx % 2
                                U[(off_i + lr) * m + (off_j + lc), col] += uf[idx]
                            if vf[idx] != 0:
                                lr, lc = idx // 2, idx % 2
                                V[(off_j + lr) * p + (off_k + lc), col] += vf[idx]
                            if wf[idx] != 0:
                                lr, lc = idx // 2, idx % 2
                                W[(off_k + lc) * n + (off_i + lr), col] += wf[idx]
                        col += 1
                else:
                    for li in range(si):
                        for lj in range(sj):
                            for lk in range(sk):
                                U[(off_i + li) * m + (off_j + lj), col] = 1.0
                                V[(off_j + lj) * p + (off_k + lk), col] = 1.0
                                W[(off_k + lk) * n + (off_i + li), col] = 1.0
                                col += 1

    assert col == total_rank

    if time.time() > deadline - 20.0:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Convert to exact fractions for dependency analysis
    U_frac = [[Fraction(int(round(x)), 1) for x in row] for row in U]
    V_frac = [[Fraction(int(round(x)), 1) for x in row] for row in V]
    W_frac = [[Fraction(int(round(x)), 1) for x in row] for row in W]

    # Check for zero columns
    zero_cols = []
    for c in range(total_rank):
        if all(U_frac[r][c] == 0 for r in range(n*m)) and \
           all(V_frac[r][c] == 0 for r in range(m*p)) and \
           all(W_frac[r][c] == 0 for r in range(p*n)):
            zero_cols.append(c)

    if zero_cols:
        keep = [c for c in range(total_rank) if c not in zero_cols]
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
        total_rank = len(keep)
        U_frac = [[U_frac[r][c] for c in keep] for r in range(n*m)]
        V_frac = [[V_frac[r][c] for c in keep] for r in range(m*p)]
        W_frac = [[W_frac[r][c] for c in keep] for r in range(p*n)]

    if time.time() > deadline - 15.0:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Check for identical columns
    col_sigs = {}
    dup_to_remove = set()
    for c in range(total_rank):
        sig = (
            tuple(U_frac[r][c] for r in range(n*m)),
            tuple(V_frac[r][c] for r in range(m*p)),
            tuple(W_frac[r][c] for r in range(p*n)),
        )
        if sig in col_sigs:
            dup_to_remove.add(c)
        else:
            col_sigs[sig] = c

    if dup_to_remove:
        keep = [c for c in range(total_rank) if c not in dup_to_remove]
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
        total_rank = len(keep)
        U_frac = [[U_frac[r][c] for c in keep] for r in range(n*m)]
        V_frac = [[V_frac[r][c] for c in keep] for r in range(m*p)]
        W_frac = [[W_frac[r][c] for c in keep] for r in range(p*n)]

    if time.time() > deadline - 10.0:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Check for linear dependencies: can any column be expressed as a linear combination of others?
    # For a column c to be redundant, there must exist coefficients alpha_r such that
    # U[:,c] = sum_r alpha_r * U[:,r], V[:,c] = sum_r alpha_r * V[:,r], W[:,c] = sum_r alpha_r * W[:,r]
    # This is a system of (n*m + m*p + p*n) equations in total_rank-1 unknowns.
    # We check this for each column c, trying to express it in terms of all other columns.
    
    # Build the combined tensor matrix T where each column is the concatenation of U_col, V_col, W_col
    # T has shape (25+25+25, total_rank) = (75, total_rank)
    T = np.zeros((75, total_rank), dtype=np.float64)
    T[:25, :] = U
    T[25:50, :] = V
    T[50:, :] = W

    # Check rank of T using SVD
    try:
        s = np.linalg.svd(T, compute_uv=False)
        # Count significant singular values
        tol = 1e-10
        rank_T = np.sum(s > tol)
        # If rank_T < total_rank, there are linear dependencies
        # But we need to find which columns are redundant and remove them while preserving exactness
    except:
        rank_T = total_rank

    # Try to find a minimal subset of columns that spans the same space
    # Use greedy forward selection
    if rank_T < total_rank:
        # Use exact fractions for the greedy selection
        # Start with an empty set, add columns that increase the rank
        selected = []
        # We'll use the exact fractions to check independence
        for c in range(total_rank):
            if time.time() > deadline - 5.0:
                break
            # Check if column c is in the span of already selected columns
            if not selected:
                selected.append(c)
                continue
            # Build matrix from selected columns and check if c is in span
            # Use exact arithmetic
            mat = []
            for r in range(75):
                row_vals = []
                for sc in selected:
                    if r < 25:
                        row_vals.append(U_frac[r][sc])
                    elif r < 50:
                        row_vals.append(V_frac[r-25][sc])
                    else:
                        row_vals.append(W_frac[r-50][sc])
                row_vals.append(T[r, c])  # This is the target value
                mat.append(row_vals)
            
            # Check if the system is consistent: is T[:,c] in span of T[:,selected]?
            # Use Gaussian elimination with fractions
            # Build augmented matrix [T[:,selected] | T[:,c]]
            aug = []
            for r in range(75):
                row = []
                for sc in selected:
                    if r < 25:
                        row.append(U_frac[r][sc])
                    elif r < 50:
                        row.append(V_frac[r-25][sc])
                    else:
                        row.append(W_frac[r-50][sc])
                if r < 25:
                    row.append(U_frac[r][c])
                elif r < 50:
                    row.append(V_frac[r-25][c])
                else:
                    row.append(W_frac[r-50][c])
                aug.append(row)
            
            # Gaussian elimination
            ncols = len(selected) + 1
            nrows = 75
            rank = 0
            for col_idx in range(ncols):
                # Find pivot
                pivot_row = -1
                for r in range(rank, nrows):
                    if aug[r][col_idx] != 0:
                        pivot_row = r
                        break
                if pivot_row == -1:
                    continue
                # Swap
                aug[rank], aug[pivot_row] = aug[pivot_row], aug[rank]
                # Eliminate
                pivot_val = aug[rank][col_idx]
                for r in range(rank+1, nrows):
                    factor = aug[r][col_idx] / pivot_val
                    for c_idx in range(col_idx, ncols):
                        aug[r][c_idx] -= factor * aug[rank][c_idx]
                rank += 1
            
            # Check if the last column (target) has a non-zero entry in a row where all pivot columns are zero
            # If the rank of the coefficient matrix equals the rank of the augmented matrix, it's consistent
            coeff_rank = 0
            for col_idx in range(len(selected)):
                found_pivot = False
                for r in range(75):
                    if aug[r][col_idx] != 0:
                        found_pivot = True
                        break
                if found_pivot:
                    coeff_rank += 1
            
            # Actually, let's just check if the system is consistent by seeing if there's a row
            # where all coefficient entries are 0 but the target entry is non-zero
            consistent = True
            for r in range(75):
                all_zero = all(aug[r][c_idx] == 0 for c_idx in range(len(selected)))
                if all_zero and aug[r][len(selected)] != 0:
                    consistent = False
                    break
            
            if not consistent:
                selected.append(c)

        if len(selected) < total_rank:
            U = U[:, selected]
            V = V[:, selected]
            W = W[:, selected]
            total_rank = len(selected)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
