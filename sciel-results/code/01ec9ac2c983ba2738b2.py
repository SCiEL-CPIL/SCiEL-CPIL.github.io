import numpy as np
import time
from collections import defaultdict

def construct():
    n, m, p = 6, 6, 6
    deadline = time.time() + 110.0

    # Strassen 2x2 block decomposition with 3x3 blocks
    # 6 = 2 * 3, so 2x2 blocks of size 3x3
    # Rank = 7 * 27 = 189
    rank = 189
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Strassen 2x2 definitions
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_defs = [
        # M0 = (A00 + A11) * (B00 + B11)
        {'A': [(0, 1.0), (3, 1.0)], 'B': [(0, 1.0), (3, 1.0)]},
        # M1 = (A10 + A11) * B00
        {'A': [(2, 1.0), (3, 1.0)], 'B': [(0, 1.0)]},
        # M2 = A00 * (B01 - B11)
        {'A': [(0, 1.0)], 'B': [(1, 1.0), (3, -1.0)]},
        # M3 = A11 * (B10 - B00)
        {'A': [(3, 1.0)], 'B': [(2, 1.0), (0, -1.0)]},
        # M4 = (A00 + A01) * B11
        {'A': [(0, 1.0), (1, 1.0)], 'B': [(3, 1.0)]},
        # M5 = (A10 - A00) * (B00 + B01)
        {'A': [(2, 1.0), (0, -1.0)], 'B': [(0, 1.0), (1, 1.0)]},
        # M6 = (A01 - A11) * (B10 + B11)
        {'A': [(1, 1.0), (3, -1.0)], 'B': [(2, 1.0), (3, 1.0)]},
    ]

    recon_coeffs = [
        # C00 = M0 + M3 - M4 + M6
        [(0, 1.0), (3, 1.0), (4, -1.0), (6, 1.0)],
        # C01 = M2 + M4
        [(2, 1.0), (4, 1.0)],
        # C10 = M1 + M3
        [(1, 1.0), (3, 1.0)],
        # C11 = M0 - M1 + M2 + M5
        [(0, 1.0), (1, -1.0), (2, 1.0), (5, 1.0)],
    ]

    bs = 3
    r_idx = 0
    for s in range(7):
        s_def = strassen_defs[s]
        for u in range(bs):
            for v in range(bs):
                for j in range(bs):
                    for block_idx, coeff_a in s_def['A']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        i_glob = br * bs + u
                        k_glob = bc * bs + v
                        u_row = i_glob * m + k_glob
                        U[u_row, r_idx] += coeff_a
                    for block_idx, coeff_b in s_def['B']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        k_glob = br * bs + v
                        j_glob = bc * bs + j
                        v_row = k_glob * p + j_glob
                        V[v_row, r_idx] += coeff_b
                    for c_block_idx in range(4):
                        gamma = 0.0
                        for (s_idx, g) in recon_coeffs[c_block_idx]:
                            if s_idx == s:
                                gamma = g
                                break
                        if gamma != 0.0:
                            br = c_block_idx // 2
                            bc = c_block_idx % 2
                            i_glob = br * bs + u
                            j_glob = bc * bs + j
                            w_row = j_glob * n + i_glob
                            W[w_row, r_idx] += gamma
                    r_idx += 1

    # Verify correctness
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                T[i * m + k, k * p + j, j * n + i] = 1.0

    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    if not np.allclose(T_recon, T, atol=1e-9):
        # Fallback to standard
        U_std = np.zeros((n * m, n * m * p), dtype=float)
        V_std = np.zeros((m * p, n * m * p), dtype=float)
        W_std = np.zeros((p * n, n * m * p), dtype=float)
        idx = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U_std[i * m + k, idx] = 1.0
                    V_std[k * p + j, idx] = 1.0
                    W_std[j * n + i, idx] = 1.0
                    idx += 1
        return {"U": U_std.tolist(), "V": V_std.tolist(), "W": W_std.tolist()}

    # Systematic rank reduction using pattern-based grouping
    # Key insight: columns with identical (U, V) patterns can be merged by adjusting W
    # More generally, if a set of columns has U vectors in a d_U-dim space and V in d_V-dim,
    # we can potentially compress to max(d_U, d_V) columns if the W structure allows.

    # Step 1: Identify columns by their nonzero pattern signatures
    # For each column, create a signature based on which entries of U and V are nonzero

    def get_signature(vec, tol=1e-12):
        """Get a hashable signature of a vector's nonzero pattern."""
        nz = np.nonzero(np.abs(vec) > tol)[0]
        return tuple(nz)

    # Group columns by (U_sig, V_sig)
    groups = defaultdict(list)
    for c in range(rank):
        u_sig = get_signature(U[:, c])
        v_sig = get_signature(V[:, c])
        groups[(u_sig, v_sig)].append(c)

    # For each group, if multiple columns share the same (U, V) pattern,
    # they can potentially be merged into fewer columns
    # If U[:, c1] = U[:, c2] and V[:, c1] = V[:, c2] exactly,
    # then W[:, c1] and W[:, c2] must satisfy:
    #   sum of W contributions must equal the original
    # If they're identical in U and V, we can just add W vectors

    # But more powerfully: if U vectors are linearly dependent AND V vectors are linearly dependent,
    # we can compress.

    # Let's use a more systematic approach:
    # For each group of columns with the same (U_sig, V_sig), check if they can be merged.

    new_U = U.copy()
    new_V = V.copy()
    new_W = W.copy()
    col_map = list(range(rank))  # maps new col index to old col index
    current_rank = rank

    # Process groups: if multiple columns have identical U and V vectors (up to scaling),
    # merge them by summing W
    merged_cols = []
    for sig, cols in groups.items():
        if len(cols) <= 1:
            continue
        # Check if all columns in this group have identical U and V vectors
        ref_u = U[:, cols[0]]
        ref_v = V[:, cols[0]]
        all_same = True
        for c in cols[1:]:
            if not np.allclose(U[:, c], ref_u, atol=1e-12):
                all_same = False
                break
            if not np.allclose(V[:, c], ref_v, atol=1e-12):
                all_same = False
                break
        if all_same:
            # Merge: keep first column, sum W contributions
            merged_W = np.zeros(p * n, dtype=float)
            for c in cols:
                merged_W += W[:, c]
            new_W[:, cols[0]] = merged_W
            # Mark other columns for removal
            for c in cols[1:]:
                merged_cols.append(c)

    if merged_cols:
        # Remove merged columns
        keep_mask = np.ones(rank, dtype=bool)
        for c in merged_cols:
            keep_mask[c] = False
        new_U = new_U[:, keep_mask]
        new_V = new_V[:, keep_mask]
        new_W = new_W[:, keep_mask]
        current_rank = int(keep_mask.sum())

    # Step 2: More aggressive reduction using linear algebra
    # For sets of columns where U vectors span a low-dim space and V vectors span a low-dim space,
    # we can potentially compress further.

    # Use the following approach:
    # 1. Find the rank of U and V matrices
    # 2. If rank(U) < current_rank or rank(V) < current_rank, there's room for reduction
    # 3. Use SVD to find a basis and project

    # Actually, the most effective approach for this structured problem:
    # The Strassen construction has a specific algebraic structure. Let's exploit it.

    # For 2x2 Strassen, the 7 terms have specific relationships. When we tile with 3x3 blocks,
    # many of the resulting columns are linearly dependent.

    # Let's use a null-space approach:
    # The tensor T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # If we have a linear dependency sum_r alpha_r * (U[:,r] ⊗ V[:,r]) = 0,
    # and the corresponding W combination is also zero, we can remove that degree of freedom.

    # More practically: try to find the minimal rank by iterative refinement.

    # Use a smarter greedy: instead of removing one column at a time with full tensor check,
    # use the structure to identify removable columns more efficiently.

    # For each column r, check if it can be expressed as a linear combination of others
    # in a way that preserves the tensor.

    # This is equivalent to: can we find alpha_s (s != r) such that:
    #   U[:, r] = sum_s alpha_s * U[:, s]  AND
    #   V[:, r] = sum_s beta_s * V[:, s]  AND
    #   W[:, r] = sum_s (alpha_s * beta_s) * W[:, s]

    # This is hard to check directly. Instead, use the following:
    # If we fix a subset S of columns to keep, the question is whether T can be represented
    # with just those columns. This is a bilinear optimization problem.

    # For now, use a faster greedy with early termination:
    # Try removing columns in order of "least contribution" (smallest norm of W)

    def fast_try_remove(col, kept_mask):
        """Try removing column 'col'. Returns True if successful."""
        if not kept_mask[col]:
            return False
        new_mask = kept_mask.copy()
        new_mask[col] = False
        if new_mask.sum() == 0:
            return False
        U_t = new_U[:, new_mask]
        V_t = new_V[:, new_mask]
        W_t = new_W[:, new_mask]
        T_t = np.einsum('ir,jr,kr->ijk', U_t, V_t, W_t)
        if np.allclose(T_t, T, atol=1e-9):
            kept_mask[:] = new_mask
            return True
        return False

    kept_mask = np.ones(current_rank, dtype=bool)
    removed_count = 0

    # Sort columns by W norm (ascending) - try removing low-contribution columns first
    w_norms = np.linalg.norm(new_W, axis=0)
    col_order = np.argsort(w_norms)

    # Pass 1: sorted by W norm
    for c in col_order:
        if time.time() > deadline - 30:
            break
        if not kept_mask[c]:
            continue
        if fast_try_remove(c, kept_mask):
            removed_count += 1

    # Pass 2: reverse order
    for c in reversed(col_order):
        if time.time() > deadline - 20:
            break
        if not kept_mask[c]:
            continue
        if fast_try_remove(c, kept_mask):
            removed_count += 1

    # Pass 3: natural order
    for c in range(current_rank):
        if time.time() > deadline - 15:
            break
        if not kept_mask[c]:
            continue
        if fast_try_remove(c, kept_mask):
            removed_count += 1

    # Final pass
    for c in range(current_rank - 1, -1, -1):
        if time.time() > deadline - 10:
            break
        if not kept_mask[c]:
            continue
        if fast_try_remove(c, kept_mask):
            removed_count += 1

    final_rank = int(kept_mask.sum())
    U_final = new_U[:, kept_mask]
    V_final = new_V[:, kept_mask]
    W_final = new_W[:, kept_mask]

    # Final verification
    T_final = np.einsum('ir,jr,kr->ijk', U_final, V_final, W_final)
    if not np.allclose(T_final, T, atol=1e-9):
        # Fallback to original Strassen
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
