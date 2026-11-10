import numpy as np
import time


def nikodym_set() -> np.ndarray:
    p = 7
    q = 49
    npts = q * q
    t0 = time.time()
    deadline = t0 + 600.0

    def idx(a, b):
        return (a % p) + 7 * (b % p)
    def unidx(i):
        return (i % p, i // p)

    # Field tables
    add = np.zeros((q, q), dtype=np.int32)
    mul = np.zeros((q, q), dtype=np.int32)
    for a1 in range(p):
        for b1 in range(p):
            for a2 in range(p):
                for b2 in range(p):
                    i1 = idx(a1, b1)
                    i2 = idx(a2, b2)
                    add[i1, i2] = idx(a1 + a2, b1 + b2)
                    mul[i1, i2] = idx(a1 * a2 + 3 * b1 * b2, a1 * b2 + a2 * b1)
    zero = idx(0, 0)
    neg = np.zeros(q, dtype=np.int32)
    for i in range(q):
        for j in range(q):
            if add[i, j] == zero:
                neg[i] = j
                break

    n_dirs = q + 1  # 50
    # point_lines: (npts, n_dirs)
    point_lines = np.zeros((npts, n_dirs), dtype=np.int32)
    for xa in range(p):
        for xb in range(p):
            x = idx(xa, xb)
            for ya in range(p):
                for yb in range(p):
                    y = idx(ya, yb)
                    pt = x * q + y
                    for m in range(q):
                        inter = add[y, mul[neg[m], x]]
                        point_lines[pt, m] = m * q + inter
                    point_lines[pt, q] = q * q + x

    def is_valid(lc):
        return lc[point_lines].min(axis=1).max() <= 1

    def build_lc(S):
        lc = np.zeros(q * n_dirs, dtype=np.int32)
        for pt in np.where(S)[0]:
            lc[point_lines[pt]] += 1
        return lc

    def greedy_add(S, lc, max_passes=4):
        order = np.random.permutation(npts)
        for _ in range(max_passes):
            if time.time() > deadline - 30:
                break
            improved = False
            for pt in order:
                if S[pt]:
                    continue
                lines = point_lines[pt]
                lc[lines] += 1
                if is_valid(lc):
                    S[pt] = True
                    improved = True
                else:
                    lc[lines] -= 1
            if not improved:
                break
        return S, lc

    # Hermitian starting S
    def hermitian_S():
        k = None
        for c in range(2, p):
            if pow(c, (p - 1) // 2, p) == p - 1:
                k = c
                break
        y_special = set()
        for u in range(p):
            for v in range(p):
                y_special.add(((u * u) % p, v))
        S = np.zeros(npts, dtype=bool)
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                        in_y = (ya, yb) in y_special
                        if in_unital and not in_y:
                            x = idx(xa, xb); y = idx(ya, yb)
                            S[x * q + y] = True
        return S

    best_S = np.zeros(npts, dtype=bool)
    best_size = 0

    def consider(S):
        nonlocal best_S, best_size
        s = S.sum()
        if s > best_size:
            best_size = s
            best_S = S.copy()

    # Greedy from empty
    for _ in range(4):
        if time.time() > deadline - 60:
            break
        S = np.zeros(npts, dtype=bool)
        lc = np.zeros(q * n_dirs, dtype=np.int32)
        S, lc = greedy_add(S, lc)
        consider(S)

    # Greedy from Hermitian
    if time.time() < deadline - 60:
        S = hermitian_S()
        lc = build_lc(S)
        S, lc = greedy_add(S, lc)
        consider(S)

    # Kick-based local search
    S = best_S.copy()
    lc = build_lc(S)
    rng = np.random.default_rng(12345)

    while time.time() < deadline - 20:
        # kick: remove k random S points
        s_now = S.sum()
        if s_now == 0:
            break
        k = int(rng.integers(1, min(6, s_now) + 1))
        idxs = np.where(S)[0]
        remove = rng.choice(idxs, size=k, replace=False)
        for pt in remove:
            S[pt] = False
            lc[point_lines[pt]] -= 1
        # greedy add
        S2, lc2 = greedy_add(S.copy(), lc.copy())
        s2 = S2.sum()
        if s2 >= best_size:
            S, lc = S2, lc2
            consider(S)
        else:
            # revert to best
            S = best_S.copy()
            lc = build_lc(S)

    return _output(best_S)


def _output(S):
    q = 49
    p = 7
    pts = []
    for i in range(q * q):
        if S[i]:
            continue
        y = i % q
        x = i // q
        xa, xb = x % p, x // p
        ya, yb = y % p, y // p
        pts.append([[xa, xb], [ya, yb]])
    return np.array(pts, dtype=np.int64)
