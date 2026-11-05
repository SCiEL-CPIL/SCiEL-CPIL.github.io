"""Matrix multiplication 3x5x6 rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]
_DEADLINE = time.time() + 145.0  # safety margin before 150s kill


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
    """Return schoolbook decomposition of rank n*m*p."""
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


def _build_block_decomposition(n, m, p, i_groups, j_groups, k_groups):
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
    """Vectorized verification using einsum."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    nm = n * m
    mp = m * p
    pn = p * n

    M_all = np.einsum('cr,ar,br->cab', W, U, V)

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


def _try_reduce_rank(U, V, W, n, m, p, max_iterations=50):
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

            ok = True
            for M in (U, V, W):
                M_others = M[:, others]
                M_target = M[:, target]
                try:
                    sol, residuals, rank_aug, sv = np.linalg.lstsq(M_others, M_target, rcond=1e-10)
                    residual = M_others @ sol - M_target
                    if np.linalg.norm(residual) > 1e-8 * max(np.linalg.norm(M_target), 1e-10):
                        ok = False
                        break
                except np.linalg.LinAlgError:
                    rank_together = np.linalg.matrix_rank(np.column_stack([M_others, M_target]))
                    rank_others = np.linalg.matrix_rank(M_others)
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
    """Generate all valid partitions of each dimension into contiguous chunks of size 1 or 2."""
    def _partitions(dim):
        results = []
        def rec(start, cur):
            if start == dim:
                results.append([tuple(g) for g in cur])
                return
            if start + 1 <= dim:
                cur.append((start, start + 1))
                rec(start + 1, cur)
                cur.pop()
            if start + 2 <= dim:
                cur.append((start, start + 2))
                rec(start + 2, cur)
                cur.pop()
        rec(0, [])
        return results

    n_parts = _partitions(n)
    m_parts = _partitions(m)
    p_parts = _partitions(p)

    all_combos = []
    for np_ in n_parts:
        for mp_ in m_parts:
            for pp_ in p_parts:
                all_combos.append((np_, mp_, pp_))

    return all_combos


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS

    # Schoolbook fallback
    sb = _schoolbook(n, m, p)
    best = sb
    best_rank = n * m * p

    # Try all block partitions
    partitions = _generate_partitions(n, m, p)

    for n_parts, m_parts, p_parts in partitions:
        if time.time() > _DEADLINE - 10:
            break
        try:
            result = _build_block_decomposition(n, m, p, n_parts, m_parts, p_parts)
            rank = len(result["U"][0]) if result["U"] else 0
            if rank < best_rank:
                if _verify_fast(result["U"], result["V"], result["W"], n, m, p):
                    best = result
                    best_rank = rank
        except Exception:
            continue

    # Try rank reduction on the best result found
    if time.time() < _DEADLINE - 5:
        try:
            U_red, V_red, W_red = _try_reduce_rank(
                best["U"], best["V"], best["W"], n, m, p, max_iterations=30
            )
            red_rank = len(U_red[0]) if U_red else 0
            if red_rank < best_rank:
                if _verify_fast(U_red, V_red, W_red, n, m, p):
                    best = {"U": U_red, "V": V_red, "W": W_red}
                    best_rank = red_rank
        except Exception:
            pass

    return best
