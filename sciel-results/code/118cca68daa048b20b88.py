import numpy as np
import time
from itertools import combinations

DIMENSIONS = [4, 5, 5]
_START = time.time()
_DEADLINE = 120.0  # seconds
_MARGIN = 10.0     # safety margin before the 150s kill


def _time_left():
    return _DEADLINE - (time.time() - _START)


def construct():
    n, m, p = DIMENSIONS  # n=4, m=5, p=5

    U_cols = []
    V_cols = []
    W_cols = []

    def add_rank1(u_vec, v_vec, w_vec):
        U_cols.append(list(u_vec))
        V_cols.append(list(v_vec))
        W_cols.append(list(w_vec))

    def add_term(i, k, j, coeff_u=1.0, coeff_v=1.0, coeff_w=1.0):
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        u_vec[i * m + k] = coeff_u
        v_vec[k * p + j] = coeff_v
        w_vec[j * n + i] = coeff_w
        add_rank1(u_vec, v_vec, w_vec)

    # ---- Part 1: 4x4x4 core via nested Strassen (7 * 7 = 49 terms) ----
    inner_strassen = [
        {'A': [(0, 0, 1), (1, 1, 1)], 'B': [(0, 0, 1), (1, 1, 1)], 'C': [(0, 0, 1), (1, 1, 1)]},
        {'A': [(1, 0, 1), (1, 1, 1)], 'B': [(0, 0, 1)], 'C': [(1, 0, 1), (1, 1, -1)]},
        {'A': [(0, 0, 1)], 'B': [(0, 1, 1), (1, 1, -1)], 'C': [(0, 1, 1), (1, 1, 1)]},
        {'A': [(1, 1, 1)], 'B': [(1, 0, 1), (0, 0, -1)], 'C': [(0, 0, 1), (1, 0, 1)]},
        {'A': [(0, 0, 1), (0, 1, 1)], 'B': [(1, 1, 1)], 'C': [(0, 0, -1), (0, 1, 1)]},
        {'A': [(1, 0, 1), (0, 0, -1)], 'B': [(0, 0, 1), (0, 1, 1)], 'C': [(1, 1, 1)]},
        {'A': [(0, 1, 1), (1, 1, -1)], 'B': [(1, 0, 1), (1, 1, 1)], 'C': [(0, 0, 1)]},
    ]

    outer_strassen = [
        {'A': {0: 1, 3: 1}, 'B': {0: 1, 3: 1}, 'C': {0: 1, 3: 1}},
        {'A': {2: 1, 3: 1}, 'B': {0: 1}, 'C': {2: 1, 3: -1}},
        {'A': {0: 1}, 'B': {1: 1, 3: -1}, 'C': {1: 1, 3: 1}},
        {'A': {3: 1}, 'B': {2: 1, 0: -1}, 'C': {0: 1, 2: 1}},
        {'A': {0: 1, 1: 1}, 'B': {3: 1}, 'C': {0: -1, 1: 1}},
        {'A': {2: 1, 0: -1}, 'B': {0: 1, 1: 1}, 'C': {3: 1}},
        {'A': {1: 1, 3: -1}, 'B': {2: 1, 3: 1}, 'C': {0: 1}},
    ]

    def a_global(block, li, lk):
        if block == 0:
            return li, lk
        elif block == 1:
            return li, lk + 2
        elif block == 2:
            return li + 2, lk
        elif block == 3:
            return li + 2, lk + 2

    def b_global(block, lk, lj):
        if block == 0:
            return lk, lj
        elif block == 1:
            return lk, lj + 2
        elif block == 2:
            return lk + 2, lj
        elif block == 3:
            return lk + 2, lj + 2

    def c_global(block, li, lj):
        if block == 0:
            return li, lj
        elif block == 1:
            return li, lj + 2
        elif block == 2:
            return li + 2, lj
        elif block == 3:
            return li + 2, lj + 2

    for prod_idx, prod in enumerate(outer_strassen):
        a_form = prod['A']
        b_form = prod['B']
        c_form = prod['C']
        for inner_idx, inner in enumerate(inner_strassen):
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            for a_block, a_sign in a_form.items():
                for li, lk, inner_sign in inner['A']:
                    gi, gk = a_global(a_block, li, lk)
                    u_vec[gi * m + gk] += a_sign * inner_sign
            for b_block, b_sign in b_form.items():
                for lk, lj, inner_sign in inner['B']:
                    gk, gj = b_global(b_block, lk, lj)
                    v_vec[gk * p + gj] += b_sign * inner_sign
            for c_block, c_sign in c_form.items():
                for li, lj, inner_sign in inner['C']:
                    gi, gj = c_global(c_block, li, lj)
                    w_vec[gj * n + gi] += c_sign * inner_sign
            add_rank1(u_vec, v_vec, w_vec)

    # ---- Part 2: 4x4x1 boundary (k in 0..3, j=4) schoolbook (16) ----
    for i in range(4):
        for k in range(4):
            add_term(i, k, 4, 1.0, 1.0, 1.0)

    # ---- Part 3: 4x1x5 boundary (k=4, j in 0..4) schoolbook (20) ----
    for i in range(4):
        for j in range(5):
            add_term(i, 4, j, 1.0, 1.0, 1.0)

    # ---- Rank-1 sparsification / merge refinement ----
    # A decomposition is exact iff for every (i,k,j) the tensor entry
    #   T[i,k,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    # equals 1.0 (the matrix-multiplication tensor).
    # We attempt to reduce rank by finding pairs (a,b) and a third term c
    # such that replacing terms {a,b} with a single term {a+b} (i.e. adding
    # their vectors) still leaves the tensor exact, OR by finding that a
    # linear combination allows cancellation. The safe, exact, cheap move:
    # if two terms a and b have proportional U, V, W (same rank-1 direction
    # up to scaling), they merge into one term.
    #
    # More generally, we search for a pair (a, b) whose combined vectors
    # (Ua+Ub, Va+Vb, Wa+Wb) produce the SAME tensor contribution as the two
    # separate terms, which happens iff the cross terms vanish:
    #   Ua*Vb*Wa + Ub*Va*Wb == 0  (as a tensor)
    # When that holds, we can replace (a,b) by (a+b) and drop one rank.

    def tensor_of(u, v, w):
        # returns a flat dict of nonzero (i,k,j) -> value for the rank-1 term
        # u: length n*m, v: length m*p, w: length p*n
        out = {}
        for ik in range(n * m):
            uu = u[ik]
            if uu == 0:
                continue
            i = ik // m
            k = ik % m
            for kj in range(m * p):
                vv = v[kj]
                if vv == 0:
                    continue
                kk = kj // p
                j = kj % p
                if kk != k:
                    continue
                ww = w[j * n + i]
                if ww == 0:
                    continue
                out[(i, k, j)] = out.get((i, k, j), 0.0) + uu * vv * ww
        return out

    def tensor_full(cols_u, cols_v, cols_w):
        out = {}
        for r in range(len(cols_u)):
            t = tensor_of(cols_u[r], cols_v[r], cols_w[r])
            for key, val in t.items():
                out[key] = out.get(key, 0.0) + val
        return out

    def is_valid(cols_u, cols_v, cols_w, target):
        t = tensor_full(cols_u, cols_v, cols_w)
        # target: dict of (i,k,j)->1.0 for all i,k,j (full MM tensor)
        for key, val in target.items():
            if abs(t.get(key, 0.0) - val) > 1e-9:
                return False
        for key, val in t.items():
            if key not in target and abs(val) > 1e-9:
                return False
        return True

    # Build the target tensor: every (i,k,j) entry is 1.0
    target = {}
    for i in range(n):
        for k in range(m):
            for j in range(p):
                target[(i, k, j)] = 1.0

    # Precompute per-term tensors for speed
    term_tensors = [tensor_of(u, v, w) for u, v, w in zip(U_cols, V_cols, W_cols)]

    def try_merge(a, b):
        # Check whether cross terms cancel so that (a+b) replaces {a,b}
        # Cross tensor = Ua*Vb*Wa + Ub*Va*Wb
        cross = {}
        for key, val in term_tensors[a].items():
            pass
        # Compute Ua*Vb*Wa
        ua, va, wa = U_cols[a], V_cols[a], W_cols[a]
        ub, vb, wb = U_cols[b], V_cols[b], W_cols[b]
        cross = {}
        for ik in range(n * m):
            uav = ua[ik]
            ubv = ub[ik]
            if uav == 0 and ubv == 0:
                continue
            i = ik // m
            k = ik % m
            for kj in range(m * p):
                vv_a = va[kj]
                vv_b = vb[kj]
                kk = kj // p
                j = kj % p
                if kk != k:
                    continue
                wv_a = wa[j * n + i]
                wv_b = wb[j * n + i]
                c = uav * vv_b * wv_a + ubv * vv_a * wv_b
                if abs(c) > 1e-12:
                    cross[(i, k, j)] = c
        # If cross is all zero, the merge is exact
        return all(abs(v) <= 1e-12 for v in cross.values())

    # Iteratively attempt merges while time allows.
    # We work on copies so we can validate globally.
    cur_u = [list(x) for x in U_cols]
    cur_v = [list(x) for x in V_cols]
    cur_w = [list(x) for x in W_cols]
    cur_t = [tensor_of(u, v, w) for u, v, w in zip(cur_u, cur_v, cur_w)]

    def add_vecs(a, b, size):
        return [a[t] + b[t] for t in range(size)]

    merged_any = True
    while merged_any and _time_left() > _MARGIN:
        merged_any = False
        R = len(cur_u)
        for a in range(R):
            if _time_left() < _MARGIN:
                break
            for b in range(a + 1, R):
                # quick support check: merge only helps if cross cancels
                if not try_merge(a, b):
                    continue
                # Perform the merge tentatively
                new_u = add_vecs(cur_u[a], cur_u[b], n * m)
                new_v = add_vecs(cur_v[a], cur_v[b], m * p)
                new_w = add_vecs(cur_w[a], cur_w[b], p * n)
                # Build candidate list without a and b, with the merged term
                cand_u = []
                cand_v = []
                cand_w = []
                for r in range(R):
                    if r == a or r == b:
                        continue
                    cand_u.append(cur_u[r])
                    cand_v.append(cur_v[r])
                    cand_w.append(cur_w[r])
                cand_u.append(new_u)
                cand_v.append(new_v)
                cand_w.append(new_w)
                # Validate globally (exactness)
                if is_valid(cand_u, cand_v, cand_w, target):
                    cur_u = cand_u
                    cur_v = cand_v
                    cur_w = cand_w
                    cur_t = [tensor_of(u, v, w) for u, v, w in zip(cur_u, cur_v, cur_w)]
                    merged_any = True
                    break
            if merged_any:
                break

    U_cols = cur_u
    V_cols = cur_v
    W_cols = cur_w

    rank = len(U_cols)
    if rank == 0:
        add_term(0, 0, 0, 1.0, 1.0, 1.0)
        rank = 1

    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
