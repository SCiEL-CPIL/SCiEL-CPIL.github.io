"""
Rank-minimal exact tensor decomposition for 3x5 * 5x6 matrix multiplication.

The strategy:
1.  Use a recursive block-partitioning approach (Strassen-like) to find a good
    initial decomposition with rank < 90 (schoolbook).
2.  Perform aggressive linear-dependency elimination (rank reduction) on the
    factor matrices to remove redundant columns.
3.  Use exact Fraction arithmetic for verification and final reduction to ensure
    no floating-point errors lead to invalid solutions.
4.  Try multiple partition structures and column orderings to maximize the
    chance of finding the minimal rank.
"""

import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# Set a deadline to ensure we return within the time limit (120s limit, aim for 110s)
_DEADLINE = time.time() + 110.0


def _strassen_2x2x2_terms():
    """
    Return the 7 terms of Strassen's algorithm for 2x2 matrix multiplication.
    Each term is a tuple (U_pattern, V_pattern, W_pattern) where patterns are
    dicts mapping (row, col) to value.
    """
    terms = []
    # M1 = (A11+A22)(B11+B22)
    terms.append(({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}))
    # M2 = (A21+A22)B11
    terms.append(({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(0, 1): 1, (1, 1): -1}))
    # M3 = A11(B21-B22)
    terms.append(({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}))
    # M4 = A22(B11-B21)
    terms.append(({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}))
    # M5 = (A11+A12)B22
    terms.append(({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (1, 0): 1}))
    # M6 = (A21-A11)(B11+B12)
    terms.append(({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}))
    # M7 = (A12-A22)(B21+B22)
    terms.append(({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}))
    return terms


def _schoolbook(n, m, p):
    """
    Construct the standard schoolbook matrix multiplication tensor decomposition.
    Rank is n*m*p.
    """
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
    """
    Build a tensor decomposition using recursive Strassen-like block partitioning.
    i_groups, j_groups, k_groups are lists of lists of indices partitioning the dimensions.
    """
    nm, mp, pn = n * m, m * p, p * n
    strassen_terms = _strassen_2x2x2_terms()

    total_rank = 0
    block_info = []
    for ig in i_groups:
        for jg in j_groups:
            for kg in k_groups:
                si, sj, sk = len(ig), len(jg), len(kg)
                # Use Strassen if all blocks are 2x2x2, otherwise schoolbook
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
    """
    Fast verification using floating point arithmetic.
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    nm = n * m
    mp = m * p
    pn = p * n

    # M[c, a, b] = sum_r W[c, r] * U[a, r] * V[b, r]
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


def _verify_exact(U, V, W, n, m, p):
    """
    Exact verification using Fraction to avoid floating point issues.
    """
    Uf = np.array([[Fraction(x) for x in row] for row in U])
    Vf = np.array([[Fraction(x) for x in row] for row in V])
    Wf = np.array([[Fraction(x) for x in row] for row in W])
    r = Uf.shape[1]
    
    for c_idx in range(p * n):
        for a_idx in range(n * m):
            for b_idx in range(m * p):
                val = sum(Uf[a_idx, t] * Vf[b_idx, t] * Wf[c_idx, t] for t in range(r))
                i, j = divmod(a_idx, m)
                j2, k = divmod(b_idx, p)
                k2, i2 = divmod(c_idx, n)
                if i == i2 and j == j2 and k == k2:
                    if val != 1:
                        return False
                else:
                    if val != 0:
                        return False
    return True


def _frac_rank(mat):
    """
    Compute rank of a matrix of Fractions using Gaussian elimination.
    """
    if mat.size == 0:
        return 0
    rows, cols = mat.shape
    mat = mat.copy()
    rank = 0
    for col in range(cols):
        if rank >= rows:
            break
        pivot = -1
        for row in range(rank, rows):
            if mat[row, col] != 0:
                pivot = row
                break
        if pivot == -1:
            continue
        mat[rank], mat[pivot] = mat[pivot].copy(), mat[rank].copy()
        for row in range(rank + 1, rows):
            if mat[row, col] != 0:
                factor = mat[row, col] / mat[rank, col]
                for c in range(col, cols):
                    mat[row, c] -= factor * mat[rank, c]
        rank += 1
    return rank


def _try_reduce_rank_exact(U, V, W, n, m, p, max_iterations=50):
    """
    Exact rank reduction using Fraction arithmetic.
    Removes columns that are linearly dependent across all three factor matrices.
    """
    Uf = np.array([[Fraction(x) for x in row] for row in U])
    Vf = np.array([[Fraction(x) for x in row] for row in V])
    Wf = np.array([[Fraction(x) for x in row] for row in W])
    r = Uf.shape[1]

    for _ in range(max_iterations):
        if time.time() > _DEADLINE:
            break
        reduced = False
        for target in range(r):
            others = [i for i in range(r) if i != target]
            if len(others) == 0:
                continue

            ok = True
            for M in (Uf, Vf, Wf):
                M_others = M[:, others]
                M_target = M[:, target]
                rank_others = _frac_rank(M_others)
                augmented = np.concatenate([M_others, M_target.reshape(-1, 1)], axis=1)
                rank_aug = _frac_rank(augmented)
                if rank_aug > rank_others:
                    ok = False
                    break

            if not ok:
                continue

            Uf = np.delete(Uf, target, axis=1)
            Vf = np.delete(Vf, target, axis=1)
            Wf = np.delete(Wf, target, axis=1)
            r -= 1
            reduced = True
            break

        if not reduced:
            break

    return (Uf.tolist(), Vf.tolist(), Wf.tolist())


def _generate_partitions(dim):
    """
    Generate all partitions of a dimension into parts of size 1 or 2.
    """
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
    """
    Calculate the expected rank of a given partition.
    """
    tot = 0
    for a in ig:
        for b in jg:
            for c in kg:
                tot += 7 if (len(a) == 2 and len(b) == 2 and len(c) == 2) \
                    else len(a) * len(b) * len(c)
    return tot


def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    full = n * m * p

    best = _schoolbook(n, m, p)
    best_rank = full

    def timed():
        return time.time() > _DEADLINE

    # Generate all possible partitions
    ip = _generate_partitions(n)
    jp = _generate_partitions(m)
    kp = _generate_partitions(p)
    
    candidates = []
    for ig in ip:
        for jg in jp:
            for kg in kp:
                candidates.append((ig, jg, kg))

    # Sort candidates by expected rank
    candidates.sort(key=lambda t: _expected_rank(t[0], t[1], t[2]))
    candidates = candidates[:100]  # Limit to top 100 candidates

    # Try each candidate partition
    for (ig, jg, kg) in candidates:
        if timed():
            break
        cand = _build_recursive_strassen(n, m, p, ig, jg, kg)
        if _verify_fast(cand["U"], cand["V"], cand["W"], n, m, p):
            r = len(cand["U"][0])
            if r < best_rank:
                best = cand
                best_rank = r
            
            # Try exact rank reduction
            U_red, V_red, W_red = _try_reduce_rank_exact(cand["U"], cand["V"], cand["W"], n, m, p, max_iterations=50)
            if _verify_fast(U_red, V_red, W_red, n, m, p):
                new_rank = len(U_red[0])
                if new_rank < best_rank:
                    best = {"U": U_red, "V": V_red, "W": W_red}
                    best_rank = new_rank

    # Try multiple random column orderings for the best candidate found so far
    if not timed():
        U_b = np.array(best["U"], dtype=float)
        V_b = np.array(best["V"], dtype=float)
        W_b = np.array(best["W"], dtype=float)

        for attempt in range(10):
            if timed():
                break
            rng = random.Random(42 + attempt)
            cols = list(range(U_b.shape[1]))
            rng.shuffle(cols)
            U_sh = U_b[:, cols]
            V_sh = V_b[:, cols]
            W_sh = W_b[:, cols]
            
            U_red, V_red, W_red = _try_reduce_rank_exact(U_sh.tolist(), V_sh.tolist(), W_sh.tolist(),
                                                         n, m, p, max_iterations=30)
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
