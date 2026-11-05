"""Search for a rank-minimal exact tensor decomposition of 3x5 * 5x6 multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 110.0


def _strassen_2x2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication."""
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
    """Build a decomposition by splitting each dimension into groups."""
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
    """Fast verification: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1"""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)

    # For each (i,j,k): dot(U3[i,j,:], V3[j,k,:] * W3[k,i,:])
    # Vectorize over k for fixed (i,j):
    for i in range(n):
        for j in range(m):
            # V3[j,:,:] has shape (p, r), W3[:,:,i] has shape (p, n) -> W3[:,i,i] is (p,)
            # Actually W3[k,i,:] for fixed i: W3[:, i, :] has shape (p, r)
            # So V3[j,k,:] * W3[k,i,:] summed over k:
            # = sum_k V3[j,k,:] * W3[k,i,:]  (element-wise, then sum over k)
            # This is: (V3[j] * W3[:, i]).sum(axis=0)  where V3[j] is (p,r), W3[:,i] is (p,r)
            prod_sum = (V3[j] * W3[:, i]).sum(axis=0)  # shape (r,)
            # Now T[i,j,k] = dot(U3[i,j,:], V3[j,k,:] * W3[k,i,:])
            # But we need per-k, not summed over k. Let me redo.
            # T[i,j,k] = dot(U3[i,j,:], V3[j,k,:] * W3[k,i,:])
            # For fixed (i,j), this is a vector of length p.
            # V3[j] is (p, r), W3[:, i] is (p, r)
            # element-wise product: (p, r), then dot with U3[i,j,:] (r,) gives (p,)
            T_ij = (V3[j] * W3[:, i]).T @ U3[i, j]  # shape (p,)
            if not np.allclose(T_ij, 1.0, atol=1e-9):
                return False
    return True


def _verify_fast2(U, V, W, n, m, p):
    """Even faster verification using matrix operations."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)

    # For each i: T[i] should be all ones of shape (m, p)
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # For fixed i: T_i[j,k] = sum_c U3[i,j,c] * (V3 * W3[:,:,i])[j,k,c]
    # where (V3 * W3[:,:,i])[j,k,c] = V3[j,k,c] * W3[k,i,c]
    for i in range(n):
        # V3 * W3[:,:,i] broadcast: V3 is (m,p,r), W3[:,:,i] is (p,n,r) -> need W3[:,i,:] which is (p,r)
        # Actually W3[k,i,:] for all k: W3[:, i, :] has shape (p, r)
        # We want V3[j,k,c] * W3[k,i,c] for all j,k,c
        # = V3 * W3[:, i, :][None, :, :]  -> (m, p, r)
        VW = V3 * W3[:, i, :][None, :, :]  # (m, p, r)
        # T_i[j,k] = sum_c U3[i,j,c] * VW[j,k,c]
        # = (U3[i] * VW).sum(axis=2)  where U3[i] is (m,r), need to broadcast
        # U3[i][:, :, None] * VW -> (m, p, r), sum over r
        T_i = (U3[i][:, :, None] * VW).sum(axis=2)  # (m, p)
        if not np.allclose(T_i, 1.0, atol=1e-9):
            return False
    return True


def _try_reduce_rank(U, V, W, n, m, p, max_iterations=200, use_tensor_check=False):
    """Try to reduce rank by removing columns. If use_tensor_check, verify tensor after removal."""
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    r = U.shape[1]

    for _ in range(max_iterations):
        if time.time() > _DEADLINE:
            break
        reduced = False
        for target in range(r):
            if time.time() > _DEADLINE:
                break
            others = [i for i in range(r) if i != target]
            if len(others) == 0:
                continue

            if use_tensor_check:
                # Try removing and check if tensor is still valid
                U_try = np.delete(U, target, axis=1)
                V_try = np.delete(V, target, axis=1)
                W_try = np.delete(W, target, axis=1)
                if _verify_fast2(U_try, V_try, W_try, n, m, p):
                    U, V, W = U_try, V_try, W_try
                    r -= 1
                    reduced = True
                    break
            else:
                # Column rank check: column must be in span of others in all 3 matrices
                ok = True
                for M in (U, V, W):
                    M_others = M[:, others]
                    M_target = M[:, target]
                    r1 = np.linalg.matrix_rank(M_others, tol=1e-10)
                    r2 = np.linalg.matrix_rank(np.column_stack([M_others, M_target]), tol=1e-10)
                    if r2 > r1:
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


def _generate_partitions(dim):
    """Generate all ways to partition [0..dim) into contiguous chunks of size 1 or 2."""
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


def _expected_rank(ig, jg, kg):
    tot = 0
    for a in ig:
        for b in jg:
            for c in kg:
                tot += 7 if (len(a) == 2 and len(b) == 2 and len(c) == 2) \
                    else len(a) * len(b) * len(c)
    return tot


def _polish(U, V, W, n, m, p):
    """Snap values to nearest integer or half-integer if close."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for M in (U, V, W):
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                snapped = round(v * 2) / 2
                if abs(v - snapped) < 1e-8:
                    M[i, j] = snapped
    return U.tolist(), V.tolist(), W.tolist()


def _reduce_with_tensor_check(U, V, W, n, m, p, max_rounds=50):
    """Aggressive reduction using full tensor verification after each removal."""
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    r = U.shape[1]

    for _ in range(max_rounds):
        if time.time() > _DEADLINE:
            break
        reduced = False
        # Try each column, prefer removing ones with many zeros
        col_spareness = []
        for c in range(r):
            zeros = (np.abs(U[:, c]) < 1e-10).sum() + \
                    (np.abs(V[:, c]) < 1e-10).sum() + \
                    (np.abs(W[:, c]) < 1e-10).sum()
            col_spareness.append((zeros, c))
        col_spareness.sort(reverse=True)

        for _, target in col_spareness:
            if time.time() > _DEADLINE:
                break
            others = [i for i in range(r) if i != target]
            if len(others) == 0:
                break
            U_try = np.delete(U, target, axis=1)
            V_try = np.delete(V, target, axis=1)
            W_try = np.delete(W, target, axis=1)
            if _verify_fast2(U_try, V_try, W_try, n, m, p):
                U, V, W = U_try, V_try, W_try
                r -= 1
                reduced = True
                break

        if not reduced:
            break

    return U.tolist(), V.tolist(), W.tolist()


def _search_over_field(U, V, W, n, m, p, primes=(2, 3, 5, 7, 11)):
    """Try to find a lower-rank decomposition over finite fields, then lift."""
    best_U, best_V, best_W = U, V, W
    best_rank = len(U[0]) if isinstance(U, list) else U.shape[1]

    for prime in primes:
        if time.time() > _DEADLINE:
            break
        # Reduce mod prime
        U_p = np.array(U, dtype=float) % prime
        V_p = np.array(V, dtype=float) % prime
        W_p = np.array(W, dtype=float) % prime

        # Try removing columns over the field
        U_f = U_p.copy()
        V_f = V_p.copy()
        W_f = W_p.copy()
        r = U_f.shape[1]

        for _ in range(100):
            if time.time() > _DEADLINE:
                break
            reduced = False
            for target in range(r):
                if time.time() > _DEADLINE:
                    break
                others = [i for i in range(r) if i != target]
                if len(others) == 0:
                    break
                # Check if column is in span over F_p
                ok = True
                for M in (U_f, V_f, W_f):
                    M_others = M[:, others].astype(float)
                    M_target = M[:, target].astype(float)
                    # Over F_p, use rank over reals (since p is small, this works)
                    r1 = np.linalg.matrix_rank(M_others, tol=1e-10)
                    r2 = np.linalg.matrix_rank(np.column_stack([M_others, M_target]), tol=1e-10)
                    if r2 > r1:
                        ok = False
                        break
                if not ok:
                    continue
                U_f = np.delete(U_f, target, axis=1)
                V_f = np.delete(V_f, target, axis=1)
                W_f = np.delete(W_f, target, axis=1)
                r -= 1
                reduced = True
                break
            if not reduced:
                break

        if r < best_rank:
            # Lift back: the structure suggests which columns to keep
            # Find which original columns correspond
            # This is complex; just record the rank
            pass

    return best_U, best_V, best_W


def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    full = n * m * p

    best = _schoolbook(n, m, p)
    best_rank = full

    def timed():
        return time.time() > _DEADLINE

    # Generate all partitions
    ip = _generate_partitions(n)
    jp = _generate_partitions(m)
    kp = _generate_partitions(p)

    # Build candidate list and sort by expected rank
    candidates = []
    for ig in ip:
        for jg in jp:
            for kg in kp:
                candidates.append((ig, jg, kg))

    candidates.sort(key=lambda t: _expected_rank(t[0], t[1], t[2]))

    # Strategy 1: Try all partition combinations, keep best valid
    for (ig, jg, kg) in candidates:
        if timed():
            break
        cand = _build_recursive_strassen(n, m, p, ig, jg, kg)
        if _verify_fast2(cand["U"], cand["V"], cand["W"], n, m, p):
            r = len(cand["U"][0])
            if r < best_rank:
                best = cand
                best_rank = r
            # Try column-rank reduction first (fast)
            U_red, V_red, W_red = _try_reduce_rank(cand["U"], cand["V"], cand["W"], n, m, p, max_iterations=100, use_tensor_check=False)
            if _verify_fast2(U_red, V_red, W_red, n, m, p):
                new_rank = len(U_red[0])
                if new_rank < best_rank:
                    best = {"U": U_red, "V": V_red, "W": W_red}
                    best_rank = new_rank

    # Strategy 2: Tensor-check reduction on the best (slower but more powerful)
    if not timed():
        U_red, V_red, W_red = _reduce_with_tensor_check(best["U"], best["V"], best["W"], n, m, p, max_rounds=80)
        if _verify_fast2(U_red, V_red, W_red, n, m, p):
            new_rank = len(U_red[0])
            if new_rank < best_rank:
                best = {"U": U_red, "V": V_red, "W": W_red}
                best_rank = new_rank

    # Strategy 3: Polish
    if not timed():
        U_pol, V_pol, W_pol = _polish(best["U"], best["V"], best["W"], n, m, p)
        if _verify_fast2(U_pol, V_pol, W_pol, n, m, p):
            best = {"U": U_pol, "V": V_pol, "W": W_pol}
            # More reduction after polishing
            if not timed():
                U_red, V_red, W_red = _reduce_with_tensor_check(best["U"], best["V"], best["W"], n, m, p, max_rounds=50)
                if _verify_fast2(U_red, V_red, W_red, n, m, p):
                    new_rank = len(U_red[0])
                    if new_rank < best_rank:
                        best = {"U": U_red, "V": V_red, "W": W_red}
                        best_rank = new_rank

    # Strategy 4: Try random perturbations + reduction
    if not timed():
        for trial in range(5):
            if timed():
                break
            # Start from best, add random small perturbations to some columns
            U_r = np.array(best["U"], dtype=float).copy()
            V_r = np.array(best["V"], dtype=float).copy()
            W_r = np.array(best["W"], dtype=float).copy()
            r = U_r.shape[1]
            # Perturb a few random entries
            for _ in range(3):
                for M in (U_r, V_r, W_r):
                    idx = random.randint(0, M.shape[0] - 1)
                    jdx = random.randint(0, M.shape[1] - 1)
                    M[idx, jdx] += random.choice([-1, 1]) * 0.5
            # Snap and verify
            U_s, V_s, W_s = _polish(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p)
            if _verify_fast2(U_s, V_s, W_s, n, m, p):
                U_red, V_red, W_red = _reduce_with_tensor_check(U_s, V_s, W_s, n, m, p, max_rounds=30)
                if _verify_fast2(U_red, V_red, W_red, n, m, p):
                    new_rank = len(U_red[0])
                    if new_rank < best_rank:
                        best = {"U": U_red, "V": V_red, "W": W_red}
                        best_rank = new_rank

    # Final verification
    if not _verify_fast2(best["U"], best["V"], best["W"], n, m, p):
        best = _schoolbook(n, m, p)

    return best
# EVOLVE-BLOCK-END
