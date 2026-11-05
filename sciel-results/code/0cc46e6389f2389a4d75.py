"""Rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.

Strategy:
  * Build a rank-44 base using 4 Strassen(2x2x2) blocks for the 2x2 core
    and schoolbook for the 2x1 tail.
  * Use slice-rank structural analysis to guide targeted column elimination.
  * Try single, pair, and triple column elimination with integer/half-integer rounding.
  * Group columns by output slice k for structural elimination.
  * Exact validation at each step.
"""
import time
import random
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 150s kill

    def _ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    num_terms = n * m * p  # 48

    # ---- Schoolbook fallback (rank 48) ----
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # ---- Exact validation ----
    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                u_row = U[i * m + a]
                for k in range(p):
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    acc = 0
                    for c in range(rank):
                        uv = u_row[c] * v_row[c]
                        if uv:
                            acc += uv * w_row[c]
                    if acc != 1:
                        return False
        return True

    # ---- Strassen 2x2x2 (rank 7) ----
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]

    # ---- Build rank-44 base ----
    def _build_base():
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        # Part 1: 4 Strassen blocks for A1(2x2) * B1(2x8)
        for block_idx in range(4):
            k0 = block_idx * 2
            for r in range(7):
                for li in range(2):
                    for la in range(2):
                        u_val = S_U[r][li * 2 + la]
                        if u_val:
                            U[li * m + la][col] = u_val
                for la in range(2):
                    for lk in range(2):
                        v_val = S_V[r][la * 2 + lk]
                        if v_val:
                            V[la * p + (k0 + lk)][col] = v_val
                for lk in range(2):
                    for li in range(2):
                        w_val = S_W[r][lk * 2 + li]
                        if w_val:
                            W[(k0 + lk) * n + li][col] = w_val
                col += 1

        # Part 2: A2(2x1) * B2(1x8) schoolbook
        for i in range(2):
            for k in range(8):
                U[i * m + 2][col] = 1
                V[2 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

        return {"U": U, "V": V, "W": W}

    base = _build_base()
    if not _validate(base):
        return _schoolbook()

    best_candidate = base
    best_rank = 44

    U, V, W = base["U"], base["V"], base["W"]
    rank0 = len(U[0])  # 44

    # ---- Build column vectors in tensor space ----
    def _col_vector(U, V, W, c):
        vec = [0] * num_terms
        for i in range(n):
            for a in range(m):
                uv = U[i * m + a][c]
                if not uv:
                    continue
                for k in range(p):
                    v_val = V[a * p + k][c]
                    if not v_val:
                        continue
                    w_val = W[k * n + i][c]
                    if not w_val:
                        continue
                    vec[i * m * p + a * p + k] = uv * v_val * w_val
        return vec

    cols = [_col_vector(U, V, W, c) for c in range(rank0)]

    # ---- (a) Duplicate column removal ----
    seen = {}
    removable = []
    for c in range(rank0):
        key = tuple(cols[c])
        if key in seen:
            removable.append(c)
        else:
            seen[key] = c

    for c_rem in removable:
        if not _ok():
            break
        new_rank = rank0 - 1
        new_U = [[0] * new_rank for _ in range(n * m)]
        new_V = [[0] * new_rank for _ in range(m * p)]
        new_W = [[0] * new_rank for _ in range(p * n)]
        nc = 0
        for c in range(rank0):
            if c == c_rem:
                continue
            for i in range(n * m):
                new_U[i][nc] = U[i][c]
            for i in range(m * p):
                new_V[i][nc] = V[i][c]
            for i in range(p * n):
                new_W[i][nc] = W[i][c]
            nc += 1
        cand = {"U": new_U, "V": new_V, "W": new_W}
        if _validate(cand):
            best_candidate = cand
            best_rank = new_rank
            return best_candidate

    # ---- (b) Global SVD-based rank reduction ----
    M = np.zeros((num_terms, rank0), dtype=np.float64)
    for c in range(rank0):
        for r in range(num_terms):
            if cols[c][r]:
                M[r, c] = float(cols[c][r])

    try:
        sv_vals = np.linalg.svd(M, compute_uv=False)
        tol = 1e-10 * max(sv_vals) if len(sv_vals) > 0 else 1e-10
        num_rank = int(np.sum(sv_vals > tol))
    except np.linalg.LinAlgError:
        num_rank = rank0

    if num_rank < rank0:
        # Find basis columns greedily
        basis_cols = []
        for c in range(rank0):
            if len(basis_cols) >= num_rank:
                break
            if not basis_cols:
                basis_cols.append(c)
                continue
            B = M[:, basis_cols]
            b = M[:, c]
            try:
                x, residuals, rnk, sv = np.linalg.lstsq(B, b, rcond=1e-10)
                resid = np.linalg.norm(B @ x - b)
                if resid > 1e-8:
                    basis_cols.append(c)
            except np.linalg.LinAlgError:
                basis_cols.append(c)

        if len(basis_cols) < rank0:
            non_basis = [c for c in range(rank0) if c not in basis_cols]
            B = M[:, basis_cols]
            coeffs = {}
            success = True
            for c in non_basis:
                b = M[:, c]
                try:
                    x, residuals, rnk, sv = np.linalg.lstsq(B, b, rcond=1e-10)
                    x_round = np.round(x * 2) / 2.0
                    if np.allclose(B @ x_round, b, atol=1e-10):
                        coeffs[c] = x_round
                    else:
                        success = False
                        break
                except np.linalg.LinAlgError:
                    success = False
                    break

            if success:
                new_rank = len(basis_cols)
                new_U = [[0] * new_rank for _ in range(n * m)]
                new_V = [[0] * new_rank for _ in range(m * p)]
                new_W = [[0] * new_rank for _ in range(p * n)]

                for j, bc in enumerate(basis_cols):
                    for i in range(n * m):
                        new_U[i][j] = U[i][bc]
                    for i in range(m * p):
                        new_V[i][j] = V[i][bc]
                    for i in range(p * n):
                        new_W[i][j] = W[i][bc]

                for c in non_basis:
                    x_round = coeffs[c]
                    for j, bc in enumerate(basis_cols):
                        wgt = x_round[j]
                        if wgt != 0:
                            for i in range(n * m):
                                new_U[i][j] += wgt * U[i][c]
                            for i in range(m * p):
                                new_V[i][j] += wgt * V[i][c]
                            for i in range(p * n):
                                new_W[i][j] += wgt * W[i][c]

                cand = {"U": new_U, "V": new_V, "W": new_W}
                if _validate(cand):
                    best_candidate = cand
                    best_rank = new_rank
                    return best_candidate

    # ---- (c) Single-column elimination with integer and half-integer rounding ----
    def _try_eliminate_single(c):
        nonlocal best_candidate, best_rank
        kept = [c2 for c2 in range(rank0) if c2 != c]
        ncols = len(kept)

        A = np.zeros((num_terms, ncols), dtype=np.float64)
        for j, c2 in enumerate(kept):
            for r in range(num_terms):
                if cols[c2][r]:
                    A[r, j] = float(cols[c2][r])

        b = np.array([-float(x) for x in cols[c]], dtype=np.float64)

        # Check if b is in the span of A
        try:
            x, residuals, rank_sv, sv = np.linalg.lstsq(A, b, rcond=1e-10)
        except np.linalg.LinAlgError:
            return False

        if residuals and residuals[0] > 1e-8:
            return False

        # Try to find integer or half-integer coefficients
        # We check if x is close to integer or half-integer
        # If not, we might try to find a small integer vector in the null space of [A | b]
        # But for now, we stick to rounding the LS solution, as it's the most likely candidate
        # if a simple dependency exists.
        
        found = False
        for scale in [1, 2]:
            x_round = np.round(x * scale) / scale
            if not np.allclose(A @ x_round, b, atol=1e-10):
                continue
            
            # Check if coefficients are valid (they are by construction)
            new_rank = ncols
            new_U = [[0] * new_rank for _ in range(n * m)]
            new_V = [[0] * new_rank for _ in range(m * p)]
            new_W = [[0] * new_rank for _ in range(p * n)]
            for j, c2 in enumerate(kept):
                wgt = x_round[j]
                for i in range(n * m):
                    new_U[i][j] = U[i][c2] + wgt * U[i][c]
                for i in range(m * p):
                    new_V[i][j] = V[i][c2] + wgt * V[i][c]
                for i in range(p * n):
                    new_W[i][j] = W[i][c2] + wgt * W[i][c]

            cand = {"U": new_U, "V": new_V, "W": new_W}
            if _validate(cand):
                best_candidate = cand
                best_rank = new_rank
                found = True
                break
        
        if found:
            return True
        
        # If rounding failed, try a small search for integer dependencies
        # This is expensive, so we only do it if we have time
        if _ok():
            # We look for a vector w in {-2, -1, 0, 1, 2}^ncols such that A @ w == -v_c
            # This is a subset sum problem, which is NP-hard, but ncols is small (43)
            # However, the vector is sparse.
            # Instead, we can use the fact that the rank of A is likely close to ncols.
            # We can compute the null space of the augmented matrix [A | -v_c]
            # If there is a null vector, it gives a dependency.
            
            # Let's try to find a dependency using SVD of the augmented matrix
            Aug = np.hstack([A, b.reshape(-1, 1)])
            try:
                U_svd, S, Vt = np.linalg.svd(Aug, full_matrices=False)
                # The last column of Vt is the null vector if the smallest singular value is ~0
                # But we need the dependency to involve the last column (b) with coefficient -1
                # So we look for a null vector z such that z[-1] != 0
                # Then A @ z[:-1] = -z[-1] * b
                # We want A @ w = -b, so w = z[:-1] / z[-1]
                
                # Find null space vectors
                tol = 1e-8
                null_space_indices = [i for i in range(len(S)) if S[i] < tol]
                
                for idx in null_space_indices:
                    z = Vt[idx]
                    if abs(z[-1]) > 1e-6:
                        w = z[:-1] / z[-1]
                        # Check if w is close to integer or half-integer
                        for scale in [1, 2]:
                            w_round = np.round(w * scale) / scale
                            if np.allclose(A @ w_round, b, atol=1e-10):
                                new_rank = ncols
                                new_U = [[0] * new_rank for _ in range(n * m)]
                                new_V = [[0] * new_rank for _ in range(m * p)]
                                new_W = [[0] * new_rank for _ in range(p * n)]
                                for j, c2 in enumerate(kept):
                                    wgt = w_round[j]
                                    for i in range(n * m):
                                        new_U[i][j] = U[i][c2] + wgt * U[i][c]
                                    for i in range(m * p):
                                        new_V[i][j] = V[i][c2] + wgt * V[i][c]
                                    for i in range(p * n):
                                        new_W[i][j] = W[i][c2] + wgt * W[i][c]

                                cand = {"U": new_U, "V": new_V, "W": new_W}
                                if _validate(cand):
                                    best_candidate = cand
                                    best_rank = new_rank
                                    return True
            except np.linalg.LinAlgError:
                pass
        
        return False

    # Group columns by output slice k for structural ordering
    def _get_slice_grouping():
        """Group columns by which output indices they affect."""
        groups = {}  # frozenset of k values -> list of column indices
        for c in range(rank0):
            affected_ks = set()
            for i in range(n):
                for a in range(m):
                    if U[i * m + a][c] != 0:
                        for k in range(p):
                            if V[a * p + k][c] != 0 and W[k * n + i][c] != 0:
                                affected_ks.add(k)
            key = frozenset(affected_ks)
            if key not in groups:
                groups[key] = []
            groups[key].append(c)
        return groups

    slice_groups = _get_slice_grouping()
    
    # Order: try columns within same slice group first (structural locality)
    order = []
    for key, group_cols in slice_groups.items():
        order.extend(group_cols)
    random.seed(42)
    random.shuffle(order)
    for c in order:
        if not _ok():
            break
        if _try_eliminate_single(c):
            return best_candidate

    # ---- (d) Pair-column elimination with structural grouping ----
    def _try_eliminate_pair(c1, c2):
        nonlocal best_candidate, best_rank
        kept = [c for c in range(rank0) if c != c1 and c != c2]
        ncols = len(kept)

        A = np.zeros((num_terms, ncols), dtype=np.float64)
        for j, c2 in enumerate(kept):
            for r in range(num_terms):
                if cols[c2][r]:
                    A[r, j] = float(cols[c2][r])

        b = np.zeros(num_terms, dtype=np.float64)
        for r in range(num_terms):
            b[r] = -(cols[c1][r] + cols[c2][r])

        try:
            x, residuals, rank_sv, sv = np.linalg.lstsq(A, b, rcond=1e-10)
        except np.linalg.LinAlgError:
            return False

        if residuals and residuals[0] > 1e-8:
            return False

        for scale in [1, 2]:
            x_round = np.round(x * scale) / scale
            if not np.allclose(A @ x_round, b, atol=1e-10):
                continue

            new_rank = ncols
            new_U = [[0] * new_rank for _ in range(n * m)]
            new_V = [[0] * new_rank for _ in range(m * p)]
            new_W = [[0] * new_rank for _ in range(p * n)]
            for j, c2 in enumerate(kept):
                wgt = x_round[j]
                for i in range(n * m):
                    new_U[i][j] = U[i][c2] + wgt * (U[i][c1] + U[i][c2])
                for i in range(m * p):
                    new_V[i][j] = V[i][c2] + wgt * (V[i][c1] + V[i][c2])
                for i in range(p * n):
                    new_W[i][j] = W[i][c2] + wgt * (W[i][c1] + W[i][c2])

            cand = {"U": new_U, "V": new_V, "W": new_W}
            if _validate(cand):
                best_candidate = cand
                best_rank = new_rank
                return True
        return False

    # Try pairs within same slice group first, then cross-group
    tail_start = 28
    # Phase 1: pairs within same slice group (structural)
    for key, group_cols in slice_groups.items():
        if not _ok():
            break
        if len(group_cols) < 2:
            continue
        for idx1 in range(len(group_cols)):
            if not _ok():
                break
            for idx2 in range(idx1 + 1, len(group_cols)):
                if not _ok():
                    break
                c1, c2 = group_cols[idx1], group_cols[idx2]
                if _try_eliminate_pair(c1, c2):
                    return best_candidate

    # Phase 2: all pairs in tail region
    for c1 in range(tail_start, rank0):
        if not _ok():
            break
        for c2 in range(c1 + 1, rank0):
            if not _ok():
                break
            if _try_eliminate_pair(c1, c2):
                return best_candidate

    # ---- (e) Triple-column elimination ----
    def _try_eliminate_triple(c1, c2, c3):
        nonlocal best_candidate, best_rank
        kept = [c for c in range(rank0) if c not in (c1, c2, c3)]
        ncols = len(kept)

        A = np.zeros((num_terms, ncols), dtype=np.float64)
        for j, c2 in enumerate(kept):
            for r in range(num_terms):
                if cols[c2][r]:
                    A[r, j] = float(cols[c2][r])

        b = np.zeros(num_terms, dtype=np.float64)
        for r in range(num_terms):
            b[r] = -(cols[c1][r] + cols[c2][r] + cols[c3][r])

        try:
            x, residuals, rank_sv, sv = np.linalg.lstsq(A, b, rcond=1e-10)
        except np.linalg.LinAlgError:
            return False

        if residuals and residuals[0] > 1e-8:
            return False

        for scale in [1, 2]:
            x_round = np.round(x * scale) / scale
            if not np.allclose(A @ x_round, b, atol=1e-10):
                continue

            new_rank = ncols
            new_U = [[0] * new_rank for _ in range(n * m)]
            new_V = [[0] * new_rank for _ in range(m * p)]
            new_W = [[0] * new_rank for _ in range(p * n)]
            for j, c2 in enumerate(kept):
                wgt = x_round[j]
                for i in range(n * m):
                    new_U[i][j] = U[i][c2] + wgt * (U[i][c1] + U[i][c2] + U[i][c3])
                for i in range(m * p):
                    new_V[i][j] = V[i][c2] + wgt * (V[i][c1] + V[i][c2] + V[i][c3])
                for i in range(p * n):
                    new_W[i][j] = W[i][c2] + wgt * (W[i][c1] + W[i][c2] + W[i][c3])

            cand = {"U": new_U, "V": new_V, "W": new_W}
            if _validate(cand):
                best_candidate = cand
                best_rank = new_rank
                return True
        return False

    # Try triples within same slice group
    for key, group_cols in slice_groups.items():
        if not _ok():
            break
        if len(group_cols) < 3:
            continue
        for combo in combinations(group_cols, 3):
            if not _ok():
                break
            if _try_eliminate_triple(*combo):
                return best_candidate

    if best_rank < 44:
        return best_candidate
    return base

# EVOLVE-BLOCK-END
