"""Search for a rank-minimal exact tensor decomposition of 3x5 * 5x6 multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 110.0   # hard wall-clock guard (evaluator kills at 150s)


def _strassen_2x2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is (u_pat, v_pat, w_pat) mapping local index tuples to integer values.
    Convention: U index (li,lj), V index (lj,lk), W index (lk,li)."""
    terms = []
    terms.append(({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}))
    terms.append(({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(0, 1): 1, (1, 1): -1}))
    terms.append(({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}))
    terms.append(({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}))
    terms.append(({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (1, 0): 1}))
    terms.append(({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}))
    terms.append(({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}))
    return terms


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _build_recursive_strassen(n, m, p, i_groups, j_groups, k_groups):
    """Build a decomposition by splitting each dimension into groups.
    Any (2,2,2) block uses the 7-term Strassen trick; others use schoolbook."""
    nm, mp, pn = n * m, m * p, p * n
    strassen_terms = _strassen_2x2x2_terms()

    total_rank = 0
    block_info = []
    for ig in i_groups:
        for jg in j_groups:
            for kg in k_groups:
                si, sj, sk = len(ig), len(jg), len(kg)
                r = 7 if (si == 2 and sj == 2 and sk == 2) else si * sj * sk
                total_rank += r
                block_info.append((ig, jg, kg, r))

    U = [[0] * total_rank for _ in range(nm)]
    V = [[0] * total_rank for _ in range(mp)]
    W = [[0] * total_rank for _ in range(pn)]

    col = 0
    for ig, jg, kg, r in block_info:
        si, sj, sk = len(ig), len(jg), len(kg)
        if si == 2 and sj == 2 and sk == 2:
            for (u_pat, v_pat, w_pat) in strassen_terms:
                for (li, lj), val_u in u_pat.items():
                    gi, gj = ig[li], jg[lj]
                    U[gi * m + gj][col] = val_u
                for (lj, lk), val_v in v_pat.items():
                    gj, gk = jg[lj], kg[lk]
                    V[gj * p + gk][col] = val_v
                for (lk, li), val_w in w_pat.items():
                    gk, gi = kg[lk], ig[li]
                    W[gk * n + gi][col] = val_w
                col += 1
        else:
            for li in range(si):
                for lj in range(sj):
                    for lk in range(sk):
                        gi, gj, gk = ig[li], jg[lj], kg[lk]
                        U[gi * m + gj][col] = 1
                        V[gj * p + gk][col] = 1
                        W[gk * n + gi][col] = 1
                        col += 1

    return {"U": U, "V": V, "W": W}


def _verify_fast(U, V, W, n, m, p):
    """Fast verification: check that sum_c U[a,c]*V[b,c]*W[c'] = delta for all (a,b,c').
    We compute M = U @ V.T (nm x mp) for each W-row, but more efficiently:
    Build the full tensor check via: for each c in [0,pn), verify
    (U[:,c] * W[c,:] @ V.T) matches the expected pattern.
    
    Actually, the fastest approach: the tensor T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    should equal 1 iff a=i*m+j, b=j*p+k, c=k*n+i, else 0.
    
    We can verify by: for each c, compute outer = U[:, :] * W[c, :] then @ V.T
    But even faster: precompute UV = U @ V.T (nm x mp), then for each c:
    T[:, :, c] = element-wise: UV[a,b] * ... no, that's not right.
    
    Correct approach: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    = (U * W[c,:]) @ V.T  where * is element-wise on rows of U by W[c,:]
    
    So for each c: M_c = (U * W[c, :]) @ V.T, shape (nm, mp)
    Then T[:, :, c] = M_c
    
    We need: M_c[i*m+j, j*p+k] == 1 if c == k*n+i, else all entries in row i*m+j 
    except column j*p+k should be 0, and other rows should be 0.
    
    Actually simpler: just build the full 3D tensor check.
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    nm = n * m
    mp = m * p
    pn = p * n
    r = U.shape[1]
    
    # Compute T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # T = einsum('ar,br,cr->abc', U, V, W) but that's expensive.
    # Instead: for each c, M_c = (U * W[c,:]) @ V.T
    # Then check M_c against expected.
    
    # Expected: M_c[i*m+j, j*p+k] = 1 if c == k*n+i, else 0
    # For a given c = k0*n + i0:
    #   M_c[i0*m+j, j*p+k0] should be 1 for all j
    #   All other entries should be 0
    
    for c in range(pn):
        k0 = c // n
        i0 = c % n
        # M_c = (U * W[c, :]) @ V.T
        U_scaled = U * W[c, :]  # (nm, r)
        M_c = U_scaled @ V.T    # (nm, mp)
        
        # Check that M_c[i0*m+j, j*p+k0] == 1 for all j
        for j in range(m):
            a = i0 * m + j
            b = j * p + k0
            if abs(M_c[a, b] - 1.0) > 1e-9:
                return False
        
        # Check that all other entries are 0
        # Create a mask of expected nonzero positions
        # Expected nonzeros: rows i0*m..i0*m+m-1, cols j*p+k0 for j=0..m-1
        # Zero out expected positions and check rest
        M_check = M_c.copy()
        for j in range(m):
            M_check[i0 * m + j, j * p + k0] = 0.0
        if np.max(np.abs(M_check)) > 1e-9:
            return False
    
    return True


def _try_reduce_rank(U, V, W, n, m, p, max_iterations=100):
    """Try to reduce rank by removing columns that are linearly dependent
    across all three factor matrices simultaneously."""
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    r = U.shape[1]

    for _ in range(max_iterations):
        if time.time() > _DEADLINE:
            break
        reduced = False
        for target in range(r):
            others = [i for i in range(r) if i != target]
            if len(others) == 0:
                continue

            # Check each factor's target column is in the span of its others.
            ok = True
            for M in (U, V, W):
                M_others = M[:, others]
                M_target = M[:, target]
                # Use SVD-based rank check for speed
                rank_others = np.linalg.matrix_rank(M_others)
                rank_together = np.linalg.matrix_rank(np.column_stack([M_others, M_target]))
                if rank_together > rank_others:
                    ok = False
                    break
            if not ok:
                continue

            U = np.delete(U, target, axis=1)
            V = np.delete(V, target, axis=1)
            W = np.delete(W, target, axis=1)
            r -= 1
            reduced = True
            break

        if not reduced:
            break

    return U.tolist(), V.tolist(), W.tolist()


def _generate_partitions(n, m, p):
    """Generate candidate groupings for each dimension. Each dimension is split
    into contiguous groups of size 1 or 2 (we only exploit the 2x2x2 Strassen
    block, so groups larger than 2 give no benefit and are treated as schoolbook)."""
    def _partitions(dim):
        # all ways to partition [0..dim) into contiguous chunks of size 1 or 2
        results = []
        def rec(start, cur):
            if start == dim:
                results.append(list(cur))
                return
            if start + 1 <= dim:
                cur.append([start])
                rec(start + 1, cur)
                cur.pop()
            if start + 2 <= dim:
                cur.append([start, start + 1])
                rec(start + 2, cur)
                cur.pop()
        rec(0, [])
        return results

    ip = _partitions(n)
    jp = _partitions(m)
    kp = _partitions(p)
    return ip, jp, kp


def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    full = n * m * p

    best = _schoolbook(n, m, p)
    best_rank = full

    def timed():
        return time.time() > _DEADLINE

    # Strategy 1: recursive Strassen over many partitions, keep the best valid one.
    ip, jp, kp = _generate_partitions(n, m, p)
    candidates = []
    for ig in ip:
        for jg in jp:
            for kg in kp:
                candidates.append((ig, jg, kg))

    # Order candidates by expected rank ascending (prefer more 2x2x2 blocks).
    def expected_rank(ig, jg, kg):
        tot = 0
        for a in ig:
            for b in jg:
                for c in kg:
                    tot += 7 if (len(a) == 2 and len(b) == 2 and len(c) == 2) \
                        else len(a) * len(b) * len(c)
        return tot
    candidates.sort(key=lambda t: expected_rank(t[0], t[1], t[2]))

    for (ig, jg, kg) in candidates:
        if timed():
            break
        cand = _build_recursive_strassen(n, m, p, ig, jg, kg)
        if _verify_fast(cand["U"], cand["V"], cand["W"], n, m, p):
            r = len(cand["U"][0])
            if r < best_rank:
                best = cand
                best_rank = r
            # try to shave further via linear-dependency reduction
            U, V, W = _try_reduce_rank(cand["U"], cand["V"], cand["W"], n, m, p, max_iterations=80)
            if _verify_fast(U, V, W, n, m, p):
                new_rank = len(U[0])
                if new_rank < best_rank:
                    best = {"U": U, "V": V, "W": W}
                    best_rank = new_rank

    # Strategy 2: More aggressive rank reduction on the best candidate found.
    # The key insight: after Strassen-based construction, there may be additional
    # linear dependencies across the three factor matrices that weren't exploited.
    if not timed():
        U_b = np.array(best["U"], dtype=float)
        V_b = np.array(best["V"], dtype=float)
        W_b = np.array(best["W"], dtype=float)
        
        # Try multiple rounds of reduction with different orderings
        for attempt in range(5):
            if timed():
                break
            # Shuffle column order to find different dependencies
            rng = random.Random(42 + attempt)
            cols = list(range(U_b.shape[1]))
            rng.shuffle(cols)
            U_sh = U_b[:, cols]
            V_sh = V_b[:, cols]
            W_sh = W_b[:, cols]
            U_red, V_red, W_red = _try_reduce_rank(U_sh.tolist(), V_sh.tolist(), W_sh.tolist(),
                                                   n, m, p, max_iterations=30)
            if _verify_fast(U_red, V_red, W_red, n, m, p):
                new_rank = len(U_red[0])
                if new_rank < best_rank:
                    best = {"U": U_red, "V": V_red, "W": W_red}
                    best_rank = new_rank

    # Strategy 3: Try to merge pairs of columns via linear combinations.
    # If columns c1 and c2 can be combined as c1 + alpha*c2 in all three matrices
    # while preserving the tensor, we can reduce rank.
    if not timed():
        U_b = np.array(best["U"], dtype=float)
        V_b = np.array(best["V"], dtype=float)
        W_b = np.array(best["W"], dtype=float)
        r = U_b.shape[1]
        
        # Check if any pair of columns is linearly dependent in a way that
        # preserves the tensor structure
        for c1 in range(r):
            if timed():
                break
            for c2 in range(c1 + 1, r):
                if timed():
                    break
                # Check if removing c2 and adjusting c1 works
                # This is complex; skip for now and rely on _try_reduce_rank
                pass

    # Final verification; fall back to schoolbook if anything went wrong.
    if not _verify_fast(best["U"], best["V"], best["W"], n, m, p):
        best = _schoolbook(n, m, p)

    return best


# EVOLVE-BLOCK-END
