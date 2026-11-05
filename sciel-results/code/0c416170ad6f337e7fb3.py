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
    """Vectorized verification: compute all M_c at once and check against expected pattern."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    nm = n * m
    mp = m * p
    pn = p * n
    r = U.shape[1]
    
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # For each c: M_c = (U * W[c,:]) @ V.T
    # Vectorize: M_all[c, a, b] = sum_r U[a,r]*V[b,r]*W[c,r]
    # = einsum('ar,br,cr->cab', U, V, W)
    # But that's O(pn*nm*mp*r) which is fine for small dims.
    # Alternative: M_all = np.einsum('cr,ar,br->cab', W, U, V)
    M_all = np.einsum('cr,ar,br->cab', W, U, V)  # (pn, nm, mp)
    
    # Build expected tensor
    # Expected: T[k*n+i, i*m+j, j*p+k] = 1, else 0
    expected = np.zeros((pn, nm, mp), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c_idx = k * n + i
                a_idx = i * m + j
                b_idx = j * p + k
                expected[c_idx, a_idx, b_idx] = 1.0
    
    if np.max(np.abs(M_all - expected)) > 1e-9:
        return False
    return True


def _try_reduce_rank(U, V, W, n, m, p, max_iterations=100):
    """Try to reduce rank by removing columns that are linearly dependent
    across all three factor matrices simultaneously. Uses QR-based span check
    for speed."""
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

            ok = True
            for M in (U, V, W):
                M_others = M[:, others]
                M_target = M[:, target]
                # Use QR decomposition to check if target is in span of others
                try:
                    Q, R = np.linalg.qr(M_others)
                    # Project target onto column space of M_others
                    proj = Q @ (Q.T @ M_target)
                    residual = M_target - proj
                    if np.linalg.norm(residual) > 1e-8 * max(np.linalg.norm(M_target), 1e-10):
                        ok = False
                        break
                except np.linalg.LinAlgError:
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
    # Evaluate more candidates since reduction is now faster with QR
    candidates = candidates[:120]

    for (ig, jg, kg) in candidates:
        if timed():
            break
        cand = _build_recursive_strassen(n, m, p, ig, jg, kg)
        if _verify_fast(cand["U"], cand["V"], cand["W"], n, m, p):
            r = len(cand["U"][0])
            if r < best_rank:
                best = cand
                best_rank = r
            # try to shave further via linear-dependency reduction (faster now)
            U, V, W = _try_reduce_rank(cand["U"], cand["V"], cand["W"], n, m, p, max_iterations=100)
            if _verify_fast(U, V, W, n, m, p):
                new_rank = len(U[0])
                if new_rank < best_rank:
                    best = {"U": U, "V": V, "W": W}
                    best_rank = new_rank

    # Strategy 2: More aggressive rank reduction on the best candidate found.
    # With faster QR-based reduction, we can do many more passes.
    if not timed():
        U_b = np.array(best["U"], dtype=float)
        V_b = np.array(best["V"], dtype=float)
        W_b = np.array(best["W"], dtype=float)
        
        # Try multiple rounds of reduction with different orderings
        for attempt in range(20):
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
                                                   n, m, p, max_iterations=50)
            if _verify_fast(U_red, V_red, W_red, n, m, p):
                new_rank = len(U_red[0])
                if new_rank < best_rank:
                    best = {"U": U_red, "V": V_red, "W": W_red}
                    best_rank = new_rank
                    U_b = np.array(U_red, dtype=float)
                    V_b = np.array(V_red, dtype=float)
                    W_b = np.array(W_red, dtype=float)

    # Strategy 3: Additional reduction passes with different random seeds
    if not timed():
        U_b = np.array(best["U"], dtype=float)
        V_b = np.array(best["V"], dtype=float)
        W_b = np.array(best["W"], dtype=float)
        
        # More passes with different column orderings
        for attempt in range(20):
            if timed():
                break
            rng = random.Random(1000 + attempt)
            cols = list(range(U_b.shape[1]))
            rng.shuffle(cols)
            U_sh = U_b[:, cols]
            V_sh = V_b[:, cols]
            W_sh = W_b[:, cols]
            U_red, V_red, W_red = _try_reduce_rank(U_sh.tolist(), V_sh.tolist(), W_sh.tolist(),
                                                   n, m, p, max_iterations=50)
            if _verify_fast(U_red, V_red, W_red, n, m, p):
                new_rank = len(U_red[0])
                if new_rank < best_rank:
                    best = {"U": U_red, "V": V_red, "W": W_red}
                    best_rank = new_rank
                    U_b = np.array(U_red, dtype=float)
                    V_b = np.array(V_red, dtype=float)
                    W_b = np.array(W_red, dtype=float)

    # Final verification; fall back to schoolbook if anything went wrong.
    if not _verify_fast(best["U"], best["V"], best["W"], n, m, p):
        best = _schoolbook(n, m, p)

    return best


# EVOLVE-BLOCK-END
