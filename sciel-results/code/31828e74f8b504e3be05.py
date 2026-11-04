"""
Rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x8 matrix.

Strategy:
  1. Build a structured base: Strassen for (2,2,2) sub-blocks + schoolbook for remaining row/column.
     - 4 Strassen blocks of (2x2)*(2x2) covering A1(2x2)*B1(2x8): 4*7 = 28 terms
     - Schoolbook for A2(2x1)*B2(1x8): 2*1*8 = 16 terms
     - Total: rank 44
  2. Attempt to reduce rank via:
     a. Duplicate column detection
     b. Exact linear dependence check: for each column, check if its contribution
        can be replicated by adjusting other columns
     c. Pair-merging: check if col(c1) + col(c2) is a valid rank-1 tensor
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

    # ---- Exact validation using Fractions ----
    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
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

        # Compute column contribution vectors (exact, as list of Fractions)
        def _col_contrib(c):
            vec = [Fraction(0)] * num_terms
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
                        vec[i * m * p + a * p + k] += Fraction(u_val) * Fraction(v_val) * Fraction(w_val)
            return vec

        # Compute all column contributions
        cols = [_col_contrib(c) for c in range(rank0)]

        # Step 1: Check for duplicate columns
        seen = {}
        for c in range(rank0):
            key = tuple(cols[c])
            if key in seen:
                # Remove column c
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

        # Step 3: Try merging pairs - check if col(c1) + col(c2) is a valid rank-1 tensor
        # A rank-1 tensor in the matmul tensor space has the form:
        # T[i,a,k] = u[i,a] * v[a,k] * w[k,i]
        # We need to check if the sum vector can be factored this way.
        
        def _is_valid_rank1(vec):
            """Check if vec (list of Fractions, length 48) is a rank-1 matmul tensor.
            Returns (U_factors, V_factors, W_factors) as lists if valid, else None."""
            # vec[i*m*p + a*p + k] = U[i*m+a] * V[a*p+k] * W[k*n+i]
            # Find first non-zero entry to anchor
            first_nz = -1
            for idx in range(num_terms):
                if vec[idx] != 0:
                    first_nz = idx
                    break
            if first_nz == -1:
                return ([Fraction(0)]*(n*m), [Fraction(0)]*(m*p), [Fraction(0)]*(p*n))
            
            fi = first_nz // (m * p)
            rem = first_nz % (m * p)
            fa = rem // p
            fk = rem % p
            
            # Set U[fi,fa] = 1, then V[fa,fk] = vec[first_nz], W[fk,fi] = 1
            # But we need to be more careful. Let's use a systematic approach.
            # 
            # For a rank-1 tensor: T[i,a,k] = u_i_a * v_a_k * w_k_i
            # Pick a reference: T[fi,fa,fk] = u_fi_fa * v_fa_fk * w_fk_fi
            # 
            # Strategy: set u_fi_fa = 1, v_fa_fk = vec[first_nz], w_fk_fi = 1
            # Then for each (i,a): u[i,a] = T[i,a,fk] / (v[a,fk] * w[fk,i])
            # This is getting complex. Let's use a simpler check.
            
            # Alternative: check consistency using ratios
            # For rank-1: T[i1,a1,k1] * T[i2,a2,k2] = T[i1,a2,k1] * T[i2,a1,k2] when...
            # Actually, let's just try to factor it directly.
            
            # Initialize with zeros
            u = [Fraction(0)] * (n * m)
            v = [Fraction(0)] * (m * p)
            w = [Fraction(0)] * (p * n)
            
            # Anchor: set u[fi*m+fa] = 1
            u[fi * m + fa] = Fraction(1)
            # Then v[fa*p+fk] * w[fk*n+fi] = vec[first_nz]
            # Set w[fk*n+fi] = 1, so v[fa*p+fk] = vec[first_nz]
            w[fk * n + fi] = Fraction(1)
            v[fa * p + fk] = vec[first_nz]
            
            # Now determine all u[i,a] from T[i,a,fk] = u[i,a] * v[a,fk] * w[fk,i]
            # For a where v[a,fk] != 0 and i where w[fk,i] != 0:
            # u[i,a] = T[i,a,fk] / (v[a,fk] * w[fk,i])
            
            # First, determine all w[fk,i] for i in range(n) using T[fi,a,fk]
            # T[fi,a,fk] = u[fi,a] * v[a,fk] * w[fk,fi]
            # We know u[fi,fa]=1, v[fa,fk]=vec[first_nz], w[fk,fi]=1
            # For other a: T[fi,a,fk] = u[fi,a] * v[a,fk] * 1
            # This doesn't directly give us w[fk,i] for i != fi.
            
            # Let's use a different approach: use the full system of equations.
            # This is complex. Instead, let's verify by reconstruction.
            
            # Simpler approach: just check if the tensor satisfies the rank-1 property
            # by verifying that all 2x2 minors of certain matricizations are zero.
            
            # Matricization 1: M1[i*m+a, k] = T[i,a,k] -> shape (n*m) x p = 6x8
            # Matricization 2: M2[a*p+k, i] = T[i,a,k] -> shape (m*p) x n = 24x2
            # Matricization 3: M3[k*n+i, a] = T[i,a,k] -> shape (p*n) x m = 16x3
            
            # For rank-1, all these should have rank <= 1.
            # Check M2 (smallest): 24x2, rank <= 1 means all rows are multiples of each other.
            
            # Build M2 as list of rows
            M2 = []
            for idx in range(num_terms):
                i = idx // (m * p)
                rem = idx % (m * p)
                a = rem // p
                k = rem % p
                M2.append((a * p + k, i, vec[idx]))
            
            # Group by i
            rows_by_i = {}
            for (ak, i, val) in M2:
                if i not in rows_by_i:
                    rows_by_i[i] = []
                rows_by_i[i].append((ak, val))
            
            # For rank-1, for each i, the row vector over (a,k) should be
            # proportional to a fixed vector. Check that row for i=0 is proportional
            # to row for i=1.
            
            if n > 1:
                # Get row for i=0 and i=1
                row0 = dict(rows_by_i.get(0, []))
                row1 = dict(rows_by_i.get(1, []))
                
                # Find a non-zero entry in row0
                ref_ak = None
                ref_val0 = None
                for ak, val in row0.items():
                    if val != 0:
                        ref_ak = ak
                        ref_val0 = val
                        break
                
                if ref_ak is not None:
                    # Check that row1[ref_ak] / row0[ref_ak] is consistent for all entries
                    ratio = None
                    for ak in row0:
                        if ak in row1 and row0[ak] != 0:
                            r = row1[ak] / row0[ak]
                            if ratio is None:
                                ratio = r
                            elif r != ratio:
                                return None
                    
                    # Also check entries in row1 not in row0
                    for ak in row1:
                        if ak not in row0 and row1[ak] != 0:
                            return None  # Inconsistent support
            
            # Check M1: 6x8, rank <= 1
            # All rows should be proportional
            M1_rows = {}
            for idx in range(num_terms):
                i = idx // (m * p)
                rem = idx % (m * p)
                a = rem // p
                k = rem % p
                row_key = i * m + a
                if row_key not in M1_rows:
                    M1_rows[row_key] = {}
                M1_rows[row_key][k] = vec[idx]
            
            # Find reference row
            ref_row = None
            ref_key = None
            for key, row in M1_rows.items():
                if any(v != 0 for v in row.values()):
                    ref_row = row
                    ref_key = key
                    break
            
            if ref_row is not None:
                for key, row in M1_rows.items():
                    if key == ref_key:
                        continue
                    # Check proportionality
                    ratio = None
                    for k in ref_row:
                        if k in row and ref_row[k] != 0:
                            r = row[k] / ref_row[k]
                            if ratio is None:
                                ratio = r
                            elif r != ratio:
                                return None
                    for k in row:
                        if k not in ref_row and row[k] != 0:
                            return None
            
            # If all checks pass, extract factors
            # Reconstruct u, v, w
            # Use the anchored approach
            u = [Fraction(0)] * (n * m)
            v = [Fraction(0)] * (m * p)
            w = [Fraction(0)] * (p * n)
            
            # Find any non-zero entry
            nz_idx = -1
            for idx in range(num_terms):
                if vec[idx] != 0:
                    nz_idx = idx
                    break
            
            if nz_idx == -1:
                return (u, v, w)
            
            fi = nz_idx // (m * p)
            rem = nz_idx % (m * p)
            fa = rem // p
            fk = rem % p
            
            # Set u[fi,fa] = 1
            u[fi * m + fa] = Fraction(1)
            # v[fa,fk] * w[fk,fi] = vec[nz_idx]
            # Set w[fk,fi] = 1
            w[fk * n + fi] = Fraction(1)
            v[fa * p + fk] = vec[nz_idx]
            
            # Determine u[i,a] for all (i,a):
            # T[i,a,fk] = u[i,a] * v[a,fk] * w[fk,i]
            # For i=fi: T[fi,a,fk] = u[fi,a] * v[a,fk] * w[fk,fi] = u[fi,a] * v[a,fk]
            # So u[fi,a] = T[fi,a,fk] / v[a,fk] if v[a,fk] != 0
            
            for a in range(m):
                if a == fa:
                    continue
                t_val = vec[fi * m * p + a * p + fk]
                v_val = v[a * p + fk]
                if v_val != 0:
                    u[fi * m + a] = t_val / v_val
                elif t_val == 0:
                    u[fi * m + a] = Fraction(0)
                else:
                    return None  # Inconsistent
            
            # Determine w[fk,i] for i != fi:
            # T[fi,fa,fk] is already used. Use T[i,fa,fk] = u[i,fa] * v[fa,fk] * w[fk,i]
            # But we don't know u[i,fa] yet for i != fi.
            
            # Use T[i,a,fk] = u[i,a] * v[a,fk] * w[fk,i]
            # For a=fa: T[i,fa,fk] = u[i,fa] * v[fa,fk] * w[fk,i]
            # We know v[fa,fk]. Need another equation.
            
            # Use T[i,fa,k'] for k' != fk:
            # T[i,fa,k'] = u[i,fa] * v[fa,k'] * w[k',i]
            # This involves unknowns v[fa,k'] and w[k',i].
            
            # This is getting complex. Let's just verify by reconstruction.
            # For now, return None to be safe - we'll use a different approach below.
            return None

        # Better approach for pair merging: use numerical SVD to check rank-1,
        # then verify with exact arithmetic.
        def _try_merge_pair(c1, c2):
            """Try to merge columns c1 and c2 into a single column."""
            # Compute sum vector as float for SVD check
            sum_vec = np.zeros(num_terms)
            for idx in range(num_terms):
                sum_vec[idx] = float(cols[c1][idx]) + float(cols[c2][idx])
            
            # Check if rank-1 using matricizations
            M2 = sum_vec.reshape(m * p, n)  # 24x2
            if np.linalg.matrix_rank(M2, tol=1e-10) > 1:
                return None
            
            M1 = sum_vec.reshape(n * m, p)  # 6x8
            if np.linalg.matrix_rank(M1, tol=1e-10) > 1:
                return None
            
            M3 = sum_vec.reshape(p * n, m)  # 16x3
            if np.linalg.matrix_rank(M3, tol=1e-10) > 1:
                return None
            
            # It's approximately rank-1. Now extract factors numerically.
            # Use the anchored approach with floats, then round to rationals.
            nz_idx = -1
            for idx in range(num_terms):
                if abs(sum_vec[idx]) > 1e-15:
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
            
            u_f[fi * m + fa] = 1.0
            w_f[fk * n + fi] = 1.0
            v_f[fa * p + fk] = sum_vec[nz_idx]
            
            # Determine u[fi,a]
            for a in range(m):
                if a == fa:
                    continue
                t_val = sum_vec[fi * m * p + a * p + fk]
                v_val = v_f[a * p + fk]
                if abs(v_val) > 1e-15:
                    u_f[fi * m + a] = t_val / v_val
            
            # Determine v[a,fk] for a != fa using row fi
            for a in range(m):
                if a == fa:
                    continue
                t_val = sum_vec[fi * m * p + a * p + fk]
                u_val = u_f[fi * m + a]
                w_val = w_f[fk * n + fi]
                if abs(u_val * w_val) > 1e-15:
                    v_f[a * p + fk] = t_val / (u_val * w_val)
            
            # Determine w[fk,i] for i != fi using column fa
            for i in range(n):
                if i == fi:
                    continue
                t_val = sum_vec[i * m * p + fa * p + fk]
                u_val = u_f[i * m + fa]
                v_val = v_f[fa * p + fk]
                if abs(u_val * v_val) > 1e-15:
                    w_f[fk * n + i] = t_val / (u_val * v_val)
            
            # Determine remaining u[i,a] for i != fi, a != fa
            for i in range(n):
                if i == fi:
                    continue
                for a in range(m):
                    if a == fa:
                        continue
                    # Use T[i,a,fk] = u[i,a] * v[a,fk] * w[fk,i]
                    t_val = sum_vec[i * m * p + a * p + fk]
                    v_val = v_f[a * p + fk]
                    w_val = w_f[fk * n + i]
                    if abs(v_val * w_val) > 1e-15:
                        u_f[i * m + a] = t_val / (v_val * w_val)
            
            # Determine remaining v[a,k] for k != fk
            for a in range(m):
                for k in range(p):
                    if k == fk:
                        continue
                    # Use T[fi,a,k] = u[fi,a] * v[a,k] * w[k,fi]
                    t_val = sum_vec[fi * m * p + a * p + k]
                    u_val = u_f[fi * m + a]
                    # Need w[k,fi] - determine it
                    # Use T[fi,fa,k] = u[fi,fa] * v[fa,k] * w[k,fi]
                    # But we don't know v[fa,k] yet.
                    # Use T[i,fa,k] = u[i,fa] * v[fa,k] * w[k,i]
                    # This is circular.
                    
                    # Alternative: use the fact that it's rank-1, so
                    # T[i,a,k] = u[i,a] * v[a,k] * w[k,i]
                    # For fixed (i,a), as k varies, T[i,a,k] / w[k,i] should be proportional to v[a,k]
                    # This is still complex.
                    
                    # For now, skip and verify by reconstruction
                    pass
            
            # Verify by reconstruction
            max_err = 0.0
            for idx in range(num_terms):
                i = idx // (m * p)
                rem = idx % (m * p)
                a = rem // p
                k = rem % p
                recon = u_f[i * m + a] * v_f[a * p + k] * w_f[k * n + i]
                err = abs(recon - sum_vec[idx]) / max(abs(sum_vec[idx]), 1e-15)
                max_err = max(max_err, err)
            
            if max_err > 1e-8:
                return None
            
            # Round to nearest half-integer or integer
            def _round_to_rational(x):
                # Try to round to nearest integer or half-integer
                r = round(x)
                if abs(x - r) < 1e-6:
                    return int(r)
                r2 = round(2 * x) / 2
                if abs(x - r2) < 1e-6:
                    return r2
                # Try quarters
                r3 = round(4 * x) / 4
                if abs(x - r3) < 1e-6:
                    return r3
                # Try eighths
                r4 = round(8 * x) / 8
                if abs(x - r4) < 1e-6:
                    return r4
                return x  # Keep as float if can't round
            
            u_r = [_round_to_rational(x) for x in u_f]
            v_r = [_round_to_rational(x) for x in v_f]
            w_r = [_round_to_rational(x) for x in w_f]
            
            return (u_r, v_r, w_r)

        # Try all pairs
        for c1 in range(rank0):
            if not _elapsed_ok():
                break
            for c2 in range(c1 + 1, rank0):
                if not _elapsed_ok():
                    break
                
                factors = _try_merge_pair(c1, c2)
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

        return

    if _elapsed_ok():
        _try_reduce()

    if best_rank < 44:
        return best_candidate

    return base

# EVOLVE-BLOCK-END
