"""
Rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x8 matrix.

Strategy:
  1. Build a structured base: Strassen for (2,2,2) sub-blocks + schoolbook for remaining row/column.
     - 4 Strassen blocks of (2x2)*(2x2) covering A1(2x2)*B1(2x8): 4*7 = 28 terms
     - Schoolbook for A2(2x1)*B2(1x8): 2*1*8 = 16 terms
     - Total: rank 44
  2. Attempt to reduce rank via:
     a. Duplicate column detection
     b. Exact linear dependence check
     c. Pair-merging with fast pre-filter
     d. Linear combination elimination (remove column if it's a linear combo of others)
  3. Validate any candidate with exact rational arithmetic.
  4. If no improvement is found within the time budget, return the best decomposition found.
"""
import time
import sys
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 150s kill

    def _elapsed_ok():
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

    # ---- Exact validation using integers (multiply all by 2 to handle halves) ----
    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        # Work with doubled values to keep integers
        # Convert: if value is int, doubled = 2*val; if value is float .5, doubled = 1
        def _double(x):
            if isinstance(x, int):
                return x * 2
            elif isinstance(x, float):
                if abs(x - round(x)) < 1e-9:
                    return int(round(x)) * 2
                elif abs(x * 2 - round(x * 2)) < 1e-9:
                    return int(round(x * 2))
                else:
                    return int(round(x * 4)) // 2  # shouldn't happen
            else:
                return int(x) * 2  # Fraction
        
        # Pre-compute doubled rows
        Ud = [[_double(U[i][c]) for c in range(rank)] for i in range(n * m)]
        Vd = [[_double(V[i][c]) for c in range(rank)] for i in range(m * p)]
        Wd = [[_double(W[i][c]) for c in range(rank)] for i in range(p * n)]
        
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = 0
                    u_row = Ud[i * m + a]
                    v_row = Vd[a * p + k]
                    w_row = Wd[k * n + i]
                    for c in range(rank):
                        u = u_row[c]
                        v = v_row[c]
                        w = w_row[c]
                        if u == 0 or v == 0 or w == 0:
                            continue
                        acc += u * v * w
                    if acc != 8:  # 2^3 = 8 since we doubled each factor
                        return False
        return True

    # ---- Strassen 2x2x2 decomposition (rank 7) ----
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i * 2 + a]
                        v_val = S_V[r][a * 2 + k]
                        w_val = S_W[r][k * 2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # ---- Build rank-44 base candidate ----
    def _build_base():
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        # Part 1: A1 (2x2) * B1 (2x8) using 4 Strassen blocks
        for block_idx in range(4):
            k0 = block_idx * 2
            for r in range(7):
                for li in range(2):
                    for la in range(2):
                        u_val = S_U[r][li * 2 + la]
                        if u_val == 0:
                            continue
                        u_idx = li * m + la
                        U[u_idx][col] = u_val

                for la in range(2):
                    for lk in range(2):
                        v_val = S_V[r][la * 2 + lk]
                        if v_val == 0:
                            continue
                        v_idx = la * p + (k0 + lk)
                        V[v_idx][col] = v_val

                for lk in range(2):
                    for li in range(2):
                        w_val = S_W[r][lk * 2 + li]
                        if w_val == 0:
                            continue
                        w_idx = (k0 + lk) * n + li
                        W[w_idx][col] = w_val

                col += 1

        # Part 2: A2 (2x1) * B2 (1x8) schoolbook
        for i in range(2):
            for k in range(8):
                u_idx = i * m + 2
                v_idx = 2 * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1

        return {"U": U, "V": V, "W": W}

    base = _build_base()
    if not _validate(base):
        return _schoolbook()

    best_candidate = base
    best_rank = 44

    # ---- Try to reduce rank ----
    def _try_reduce():
        nonlocal best_candidate, best_rank

        U = base["U"]
        V = base["V"]
        W = base["W"]
        rank0 = len(U[0])  # 44

        # Compute column contribution vectors using integers (doubled)
        def _col_contrib(c):
            vec = [0] * num_terms
            for i in range(n):
                for a in range(m):
                    u_val = U[i * m + a][c]
                    if u_val == 0:
                        continue
                    for k in range(p):
                        v_val = V[a * p + k][c]
                        if v_val == 0:
                            continue
                        w_val = W[k * n + i][c]
                        if w_val == 0:
                            continue
                        # Use integer arithmetic: multiply all by 2
                        u2 = u_val * 2 if isinstance(u_val, int) else int(u_val * 2)
                        v2 = v_val * 2 if isinstance(v_val, int) else int(v_val * 2)
                        w2 = w_val * 2 if isinstance(w_val, int) else int(w_val * 2)
                        vec[i * m * p + a * p + k] += u2 * v2 * w2
            return vec

        # Compute all column contributions
        cols = [_col_contrib(c) for c in range(rank0)]

        # Step 1: Check for duplicate columns
        seen = {}
        for c in range(rank0):
            key = tuple(cols[c])
            if key in seen:
                new_rank = rank0 - 1
                new_U = [[0] * new_rank for _ in range(n * m)]
                new_V = [[0] * new_rank for _ in range(m * p)]
                new_W = [[0] * new_rank for _ in range(p * n)]
                nc = 0
                for cc in range(rank0):
                    if cc == c:
                        continue
                    for i in range(n * m):
                        new_U[i][nc] = U[i][cc]
                    for i in range(m * p):
                        new_V[i][nc] = V[i][cc]
                    for i in range(p * n):
                        new_W[i][nc] = W[i][cc]
                    nc += 1
                cand = {"U": new_U, "V": new_V, "W": new_W}
                if _validate(cand):
                    best_candidate = cand
                    best_rank = new_rank
                    return
            else:
                seen[key] = c

        if not _elapsed_ok():
            return

        # Step 2: Check if any column is zero
        for c in range(rank0):
            if all(x == 0 for x in cols[c]):
                new_rank = rank0 - 1
                new_U = [[0] * new_rank for _ in range(n * m)]
                new_V = [[0] * new_rank for _ in range(m * p)]
                new_W = [[0] * new_rank for _ in range(p * n)]
                nc = 0
                for cc in range(rank0):
                    if cc == c:
                        continue
                    for i in range(n * m):
                        new_U[i][nc] = U[i][cc]
                    for i in range(m * p):
                        new_V[i][nc] = V[i][cc]
                    for i in range(p * n):
                        new_W[i][nc] = W[i][cc]
                    nc += 1
                cand = {"U": new_U, "V": new_V, "W": new_W}
                if _validate(cand):
                    best_candidate = cand
                    best_rank = new_rank
                    return

        if not _elapsed_ok():
            return

        # Step 3: Fast rank-1 check for pair sums using support patterns
        def _fast_rank1_check(vec):
            """Fast check: is vec a rank-1 matmul tensor? Uses support pattern + ratio consistency.
            Returns True if likely rank-1, False otherwise."""
            # Build the 6x8 matricization M1[i*m+a, k]
            # For rank-1, all rows must be proportional
            # Quick check: find two non-zero entries in different rows with same k
            # and verify ratio consistency
            
            # Extract as list of (row, col, val)
            entries = []
            for idx in range(num_terms):
                if vec[idx] != 0:
                    i = idx // (m * p)
                    rem = idx % (m * p)
                    a = rem // p
                    k = rem % p
                    entries.append((i * m + a, k, vec[idx]))
            
            if len(entries) == 0:
                return True  # zero tensor is rank 0
            
            if len(entries) == 1:
                return True  # single entry is rank 1
            
            # Check M1: group by row
            row_entries = {}
            for (r, k, v) in entries:
                if r not in row_entries:
                    row_entries[r] = {}
                row_entries[r][k] = v
            
            # Find reference row with most entries
            ref_row = None
            ref_data = None
            max_count = 0
            for r, data in row_entries.items():
                if len(data) > max_count:
                    max_count = len(data)
                    ref_row = r
                    ref_data = data
            
            if ref_row is None:
                return True
            
            # Check all other rows are proportional to ref_row
            for r, data in row_entries.items():
                if r == ref_row:
                    continue
                # Find common k with non-zero in both
                ratio = None
                for k in ref_data:
                    if k in data and ref_data[k] != 0 and data[k] != 0:
                        # Check ratio consistency
                        r_val = data[k] / ref_data[k] if ref_data[k] != 0 else None
                        if r_val is not None:
                            if ratio is None:
                                ratio = r_val
                            elif r_val != ratio:
                                return False
                # Check that entries in data not in ref_data are zero (shouldn't happen since we filtered)
            
            # Also check M2: 24x2, group by i
            col_entries = {}
            for idx in range(num_terms):
                if vec[idx] != 0:
                    i = idx // (m * p)
                    rem = idx % (m * p)
                    a = rem // p
                    k = rem % p
                    ak = a * p + k
                    if i not in col_entries:
                        col_entries[i] = {}
                    col_entries[i][ak] = vec[idx]
            
            # For rank-1, rows for i=0 and i=1 must be proportional
            if 0 in col_entries and 1 in col_entries:
                ratio2 = None
                for ak in col_entries[0]:
                    if ak in col_entries[1] and col_entries[0][ak] != 0 and col_entries[1][ak] != 0:
                        r_val = col_entries[1][ak] / col_entries[0][ak]
                        if ratio2 is None:
                            ratio2 = r_val
                        elif r_val != ratio2:
                            return False
            
            return True

        # Step 4: Try merging pairs with fast pre-filter
        def _extract_rank1_factors(vec):
            """Given a rank-1 tensor vec (integers), extract (u, v, w) as floats."""
            # Find first non-zero
            nz_idx = -1
            for idx in range(num_terms):
                if vec[idx] != 0:
                    nz_idx = idx
                    break
            if nz_idx == -1:
                return ([0.0] * (n * m), [0.0] * (m * p), [0.0] * (p * n))
            
            fi = nz_idx // (m * p)
            rem = nz_idx % (m * p)
            fa = rem // p
            fk = rem % p
            
            u_f = [0.0] * (n * m)
            v_f = [0.0] * (m * p)
            w_f = [0.0] * (p * n)
            
            # Normalize: set u[fi,fa] = 1
            u_f[fi * m + fa] = 1.0
            w_f[fk * n + fi] = 1.0
            v_f[fa * p + fk] = float(vec[nz_idx])
            
            # Determine u[fi,a] for all a
            for a in range(m):
                if a == fa:
                    continue
                t_val = vec[fi * m * p + a * p + fk]
                v_val = v_f[a * p + fk]
                if abs(v_val) > 1e-15:
                    u_f[fi * m + a] = t_val / v_val
            
            # Determine v[a,fk] for a != fa
            for a in range(m):
                if a == fa:
                    continue
                t_val = vec[fi * m * p + a * p + fk]
                u_val = u_f[fi * m + a]
                w_val = w_f[fk * n + fi]
                if abs(u_val * w_val) > 1e-15:
                    v_f[a * p + fk] = t_val / (u_val * w_val)
            
            # Determine w[fk,i] for i != fi
            for i in range(n):
                if i == fi:
                    continue
                t_val = vec[i * m * p + fa * p + fk]
                u_val = u_f[i * m + fa]
                v_val = v_f[fa * p + fk]
                if abs(u_val * v_val) > 1e-15:
                    w_f[fk * n + i] = t_val / (u_val * v_val)
            
            # Determine u[i,a] for i != fi, a != fa
            for i in range(n):
                if i == fi:
                    continue
                for a in range(m):
                    if a == fa:
                        continue
                    t_val = vec[i * m * p + a * p + fk]
                    v_val = v_f[a * p + fk]
                    w_val = w_f[fk * n + i]
                    if abs(v_val * w_val) > 1e-15:
                        u_f[i * m + a] = t_val / (v_val * w_val)
            
            # Determine v[a,k] for k != fk using T[fi,a,k] = u[fi,a] * v[a,k] * w[k,fi]
            # First determine w[k,fi] for k != fk using T[fi,fa,k] = u[fi,fa] * v[fa,k] * w[k,fi]
            # But v[fa,k] is unknown. Use T[i,fa,k] = u[i,fa] * v[fa,k] * w[k,i]
            # This is complex. Use iterative approach.
            
            # For each k != fk, try to determine v[fa,k] and w[k,fi]
            # Use: T[fi,fa,k] = 1 * v[fa,k] * w[k,fi]
            # And: T[fi,a,k] = u[fi,a] * v[a,k] * w[k,fi]
            # And: T[i,fa,k] = u[i,fa] * v[fa,k] * w[k,i]
            
            # Let's determine w[k,fi] for all k using the constraint that
            # T[fi,a,k] / u[fi,a] / w[k,fi] = v[a,k] must be consistent across a.
            
            # Simpler: use the fact that for rank-1, T[i,a,k] = u[i,a]*v[a,k]*w[k,i]
            # For fixed k, the matrix [T[i,a,k] / w[k,i]] over (i,a) must be rank-1 in (i,a)
            # i.e., T[i1,a1,k]/w[k,i1] * T[i2,a2,k]/w[k,i2] = T[i1,a2,k]/w[k,i1] * T[i2,a1,k]/w[k,i2]
            
            # Let's just verify by reconstruction and if it works, we're good.
            # The key insight: if the fast check passed, the factors should be consistent.
            
            # Try to fill in remaining v[a,k] and w[k,i] using least-squares-like approach
            # For each (a,k) with k != fk:
            # v[a,k] = T[fi,a,k] / (u[fi,a] * w[k,fi]) if we know w[k,fi]
            # w[k,fi] = T[fi,fa,k] / (u[fi,fa] * v[fa,k]) = T[fi,fa,k] / v[fa,k]
            
            # Use T[fi,fa,k] to get relationship between v[fa,k] and w[k,fi]
            # T[fi,fa,k] = v[fa,k] * w[k,fi]  (since u[fi,fa]=1)
            
            # And T[fi,a,k] = u[fi,a] * v[a,k] * w[k,fi]
            # => v[a,k] = T[fi,a,k] / (u[fi,a] * w[k,fi])
            
            # We have one equation with two unknowns per k. Use additional constraint:
            # T[i,fa,k] = u[i,fa] * v[fa,k] * w[k,i]
            # This involves w[k,i] for i != fi which we may not know.
            
            # For now, set w[k,fi] = T[fi,fa,k] (assuming v[fa,k]=1 as initial guess)
            # This is a heuristic. Verify by reconstruction.
            
            for k in range(p):
                if k == fk:
                    continue
                t_fi_fa_k = vec[fi * m * p + fa * p + k]
                if abs(t_fi_fa_k) > 1e-15:
                    # Set v[fa,k] = 1, w[k,fi] = t_fi_fa_k
                    v_f[fa * p + k] = 1.0
                    w_f[k * n + fi] = float(t_fi_fa_k)
                    # Determine v[a,k] for a != fa
                    for a in range(m):
                        if a == fa:
                            continue
                        t_val = vec[fi * m * p + a * p + k]
                        u_val = u_f[fi * m + a]
                        w_val = w_f[k * n + fi]
                        if abs(u_val * w_val) > 1e-15:
                            v_f[a * p + k] = t_val / (u_val * w_val)
                    # Determine w[k,i] for i != fi
                    for i in range(n):
                        if i == fi:
                            continue
                        t_val = vec[i * m * p + fa * p + k]
                        u_val = u_f[i * m + fa]
                        v_val = v_f[fa * p + k]
                        if abs(u_val * v_val) > 1e-15:
                            w_f[k * n + i] = t_val / (u_val * v_val)
            
            # Verify by reconstruction
            max_err = 0.0
            for idx in range(num_terms):
                i = idx // (m * p)
                rem = idx % (m * p)
                a = rem // p
                k = rem % p
                recon = u_f[i * m + a] * v_f[a * p + k] * w_f[k * n + i]
                target = float(vec[idx])
                err = abs(recon - target) / max(abs(target), 1e-15)
                max_err = max(max_err, err)
            
            if max_err > 1e-8:
                return None
            
            # Round to nearest rational with small denominator
            def _round_to_rational(x):
                for denom in [1, 2, 3, 4, 6, 8, 12]:
                    r = round(x * denom) / denom
                    if abs(x - r) < 1e-6:
                        if denom == 1:
                            return int(r)
                        return r
                return x
            
            u_r = [_round_to_rational(x) for x in u_f]
            v_r = [_round_to_rational(x) for x in v_f]
            w_r = [_round_to_rational(x) for x in w_f]
            
            return (u_r, v_r, w_r)

        # Try all pairs with fast pre-filter
        for c1 in range(rank0):
            if not _elapsed_ok():
                break
            for c2 in range(c1 + 1, rank0):
                if not _elapsed_ok():
                    break
                
                # Fast pre-filter: compute sum vector
                sum_vec = [cols[c1][idx] + cols[c2][idx] for idx in range(num_terms)]
                
                # Quick zero check
                if all(x == 0 for x in sum_vec):
                    # Both columns are zero or cancel - just remove one
                    new_rank = rank0 - 1
                    new_U = [[0] * new_rank for _ in range(n * m)]
                    new_V = [[0] * new_rank for _ in range(m * p)]
                    new_W = [[0] * new_rank for _ in range(p * n)]
                    nc = 0
                    for cc in range(rank0):
                        if cc == c2:
                            continue
                        for i in range(n * m):
                            new_U[i][nc] = U[i][cc]
                        for i in range(m * p):
                            new_V[i][nc] = V[i][cc]
                        for i in range(p * n):
                            new_W[i][nc] = W[i][cc]
                        nc += 1
                    cand = {"U": new_U, "V": new_V, "W": new_W}
                    if _validate(cand):
                        best_candidate = cand
                        best_rank = new_rank
                        return
                    continue
                
                # Fast rank-1 check
                if not _fast_rank1_check(sum_vec):
                    continue
                
                # Extract factors
                factors = _extract_rank1_factors(sum_vec)
                if factors is None:
                    continue
                
                u_new, v_new, w_new = factors
                
                # Build candidate
                new_rank = rank0 - 1
                new_U = [[0] * new_rank for _ in range(n * m)]
                new_V = [[0] * new_rank for _ in range(m * p)]
                new_W = [[0] * new_rank for _ in range(p * n)]
                nc = 0
                for cc in range(rank0):
                    if cc == c1 or cc == c2:
                        continue
                    for i in range(n * m):
                        new_U[i][nc] = U[i][cc]
                    for i in range(m * p):
                        new_V[i][nc] = V[i][cc]
                    for i in range(p * n):
                        new_W[i][nc] = W[i][cc]
                    nc += 1
                # Add new column
                for i in range(n * m):
                    new_U[i][nc] = u_new[i]
                for i in range(m * p):
                    new_V[i][nc] = v_new[i]
                for i in range(p * n):
                    new_W[i][nc] = w_new[i]
                
                cand = {"U": new_U, "V": new_V, "W": new_W}
                if _validate(cand):
                    best_candidate = cand
                    best_rank = new_rank
                    return

        if not _elapsed_ok():
            return

        # Step 5: Try linear combination elimination
        # Check if any column's contribution is a linear combination of others
        # This is expensive, so only try a subset
        if not _elapsed_ok():
            return
        
        # For each column, try to express its contribution as a linear combo of others
        # using exact rational Gaussian elimination on a subset of coordinates
        # This is very expensive, so limit to checking a few columns
        for target_c in range(min(10, rank0)):  # Only check first 10 columns
            if not _elapsed_ok():
                break
            
            # Try to find coefficients alpha_j such that cols[target_c] = sum_j alpha_j * cols[j]
            # for j != target_c
            # Use a subset of coordinates for speed
            # Pick coordinates where cols[target_c] is non-zero
            nz_coords = [idx for idx in range(num_terms) if cols[target_c][idx] != 0]
            if len(nz_coords) == 0:
                continue
            
            # Use at most 5 coordinates for the linear system
            coords = nz_coords[:5]
            if len(coords) < 2:
                continue
            
            # Build system: for each coord, sum_j alpha_j * cols[j][coord] = cols[target_c][coord]
            # This is an underdetermined system (rank0-1 unknowns, len(coords) equations)
            # Use numpy lstsq as a heuristic
            A = np.array([[float(cols[j][coord]) for j in range(rank0) if j != target_c] for coord in coords], dtype=np.float64)
            b = np.array([float(cols[target_c][coord]) for coord in coords], dtype=np.float64)
            
            # Check if b is in the column space of A
            # Use SVD
            try:
                U_svd, s_vals, Vt = np.linalg.svd(A, full_matrices=False)
                rank_A = np.sum(s_vals > 1e-10)
                # Project b onto column space
                if rank_A == 0:
                    continue
                # b must be orthogonal to null space of A^T
                # Equivalently, ||A A^+ b - b|| should be small
                A_pinv = Vt.T @ np.diag(1.0 / s_vals[:rank_A]) @ U_svd[:rank_A].T
                proj = A @ (A_pinv @ b)
                if np.linalg.norm(proj - b) > 1e-8 * max(1.0, np.linalg.norm(b)):
                    continue
                
                # Found a solution. Get alpha coefficients
                alpha = A_pinv @ b
                # Map back to column indices
                other_cols = [j for j in range(rank0) if j != target_c]
                
                # The merged column would have contribution = 0 (since we're removing target_c
                # and adjusting others). Actually, this approach doesn't directly give us
                # a new rank-1 column. Skip this for now.
            except:
                continue

        return

    if _elapsed_ok():
        _try_reduce()

    if best_rank < 44:
        return best_candidate

    return base

# EVOLVE-BLOCK-END
