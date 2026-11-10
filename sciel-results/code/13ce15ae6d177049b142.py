# EVOLVE-BLOCK-START
import time
import numpy as np


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set N in F_q^2 with q = p^2, p = 7.

    Field: F_q = F_p[alpha], alpha^2 = k (k = smallest quadratic non-residue mod p).
    An element is the pair (a, b) for a + b*alpha, a, b in 0..p-1.
    A point is [[x_a, x_b], [y_a, y_b]].

    N is a Nikodym set iff for every point x in F_q^2 there is a direction
    v in {(1, m): m in F_q} U {(0,1)} such that the punctured line
    {x + t v : t in F_q, t != 0} is fully contained in N.

    We model the EXCLUDED set S = F_q^2 \ N and grow S as large as possible:
    the metric is |complement| = 2401 - |N| = |S|.

    Machinery (vectorized representation):
      line_count[x, d] = # of S-points on the line through x in direction d (excluding x).
      zero_counts[x]   = # of directions d with line_count[x, d] == 0.
    Validity: for every x, zero_counts[x] >= 1 (a valid direction still has an empty punctured line).
    Adding p to S is legal iff no neighbor n (in any direction) has zero_counts[n] == 1
    on a direction where line_count[n, d] == 0.

    Returns: np.ndarray of shape (k, 2, 2), int64, entries mod 7, duplicates merged.
    """
    p = 7
    q = p * p
    NPTS = q * q  # 2401

    start = time.time()
    deadline = start + 590.0  # safety margin vs 630s kill

    # --- Field F_p[alpha], alpha^2 = k ---
    k_const = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k_const = c
            break
    if k_const is None:
        k_const = 3

    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)

    def mul(a, b):
        return ((a[0] * b[0] + a[1] * b[1] * k_const) % p,
                (a[0] * b[1] + a[1] * b[0]) % p)

    def pt_to_idx(pt):
        xa, xb = pt[0]
        ya, yb = pt[1]
        return xa * 343 + xb * 49 + ya * 7 + yb

    def idx_to_pt(idx):
        xa = idx // 343
        idx %= 343
        xb = idx // 49
        idx %= 49
        ya = idx // 7
        yb = idx % 7
        return [[xa, xb], [ya, yb]]

    # --- Directions: (1, m) for all m in F_q, plus (0, 1) ---
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))
    directions.append(((0, 0), (1, 0)))
    num_dirs = len(directions)  # 50

    # --- Precompute affected[p, d, t] = index of the t-th other point on line through p in dir d ---
    affected = np.zeros((NPTS, num_dirs, q - 1), dtype=np.int32)
    for p_idx in range(NPTS):
        px, py = idx_to_pt(p_idx)[0], idx_to_pt(p_idx)[1]
        for d_idx in range(num_dirs):
            vx, vy = directions[d_idx]
            count = 0
            for ta in range(p):
                for tb in range(p):
                    if ta == 0 and tb == 0:
                        continue
                    tvx = mul((ta, tb), vx)
                    tvy = mul((ta, tb), vy)
                    nx = add(px, tvx)
                    ny = add(py, tvy)
                    affected[p_idx, d_idx, count] = pt_to_idx([[nx[0], nx[1]], [ny[0], ny[1]]])
                    count += 1

    # --- Line-count bookkeeping ---
    line_count = np.zeros((NPTS, num_dirs), dtype=np.int32)
    zero_counts = np.zeros(NPTS, dtype=np.int32)

    def recompute(S_list):
        line_count[:] = 0
        for s_idx in S_list:
            for d_idx in range(num_dirs):
                line_count[affected[s_idx, d_idx], d_idx] += 1
        zero_counts[:] = np.sum(line_count == 0, axis=1)

    def is_valid():
        return bool(np.all(zero_counts > 0))

    # --- Seed construction: excluded set S from a Hermitian unital with a y-block ---
    def make_seed_unital(y_block, coeff=2):
        S = set()
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        in_unital = (coeff * xa + ya * ya - yb * yb * k_const) % p == 0
                        in_y_block = (ya, yb) in y_block
                        if in_unital and not in_y_block:
                            S.add(pt_to_idx([[xa, xb], [ya, yb]]))
        return S

    # y-block candidates (y-coordinates whose points are KEPT in N, i.e. not excluded)
    y_candidates = []
    y_sq = set((u * u % p, v) for u in range(p) for v in range(p))
    y_candidates.append(y_sq)
    y_zero = set((0, v) for v in range(p))
    y_candidates.append(y_zero)
    y_01 = set((a, v) for a in (0, 1) for v in range(p))
    y_candidates.append(y_01)
    y_012 = set((a, v) for a in (0, 1, 2) for v in range(p))
    y_candidates.append(y_012)
    y_nonsq = set((a, v) for a in (3, 5, 6) for v in range(p))
    y_candidates.append(y_nonsq)
    y_03 = set((a, v) for a in (0, 3) for v in range(p))
    y_candidates.append(y_03)
    y_sq_full = set((a, v) for a in (0, 1, 2, 4) for v in range(p))
    y_candidates.append(y_sq_full)
    y_all = set((a, v) for a in range(p) for v in range(p))
    y_candidates.append(y_all)

    # --- Generate and evaluate seeds, keep the largest valid one ---
    best_seed = None
    best_size = -1
    for y_block in y_candidates:
        for coeff in (1, 2, 3, 4, 5, 6):
            if time.time() > deadline - 120:
                break
            S = make_seed_unital(y_block, coeff)
            recompute(S)
            if is_valid():
                if len(S) > best_size:
                    best_size = len(S)
                    best_seed = set(S)
    if best_seed is None:
        best_seed = make_seed_unital(y_sq, 2)
        recompute(best_seed)
        best_size = len(best_seed)

    S = set(best_seed)
    recompute(S)
    best_S = set(S)
    best_size = len(S)

    # --- Local search: greedy add, 1-swap, and deletion heuristic ---
    def try_add(p_idx):
        # Returns list of (n_idx, d_idx) neighbors that would go 0->1, or None if illegal.
        changes = []
        for d_idx in range(num_dirs):
            for n_idx in affected[p_idx, d_idx]:
                if line_count[n_idx, d_idx] == 0:
                    if zero_counts[n_idx] == 1:
                        return None
                    changes.append((n_idx, d_idx))
        return changes

    def apply_add(p_idx, changes):
        S.add(p_idx)
        for n_idx, d_idx in changes:
            line_count[n_idx, d_idx] += 1
            zero_counts[n_idx] -= 1

    def delete_heuristic():
        # Remove a point s from S if, for every direction d through s, the line still has
        # at least one OTHER S-point (so removing s breaks no line's coverage).
        # This shrinks |R| (tangent lines) and lets us add more points later.
        for s_idx in list(S):
            if time.time() > deadline - 60:
                return False
            removable = True
            for d_idx in range(num_dirs):
                # count of other S-points on line through s in dir d
                cnt = 0
                for n_idx in affected[s_idx, d_idx]:
                    if n_idx in S:
                        cnt += 1
                if cnt == 0:
                    removable = False
                    break
            if removable:
                # remove s
                for d_idx in range(num_dirs):
                    for n_idx in affected[s_idx, d_idx]:
                        line_count[n_idx, d_idx] -= 1
                        if line_count[n_idx, d_idx] == 0:
                            zero_counts[n_idx] += 1
                S.discard(s_idx)
                return True
        return False

    np.random.seed(12345)
    outside = [i for i in range(NPTS) if i not in S]

    while time.time() < deadline - 60:
        # Phase A: greedy add pass
        np.random.shuffle(outside)
        progress = False
        for p_idx in outside:
            if time.time() > deadline - 60:
                break
            if p_idx in S:
                continue
            changes = try_add(p_idx)
            if changes is not None:
                apply_add(p_idx, changes)
                outside.remove(p_idx)
                progress = True
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
        # Phase B: deletion heuristic (frees up coverage to allow more adds)
        if delete_heuristic():
            outside = [i for i in range(NPTS) if i not in S]
            progress = True
        # Phase C: 1-swap attempts (remove r, add a)
        if time.time() < deadline - 90:
            for _ in range(2000):
                if time.time() > deadline - 60:
                    break
                if not outside:
                    break
                r = np.random.choice(list(S))
                a = np.random.choice(outside)
                # simulate remove r
                ok = True
                rem_changes = []  # (n_idx, d_idx) that go 1->0 on removal
                for d_idx in range(num_dirs):
                    for n_idx in affected[r, d_idx]:
                        if line_count[n_idx, d_idx] == 1:
                            rem_changes.append((n_idx, d_idx))
                # after removal, zero_counts would increase on rem_changes neighbors
                # then add a: need to check combined validity
                # do it concretely but roll back on failure
                # apply removal
                for n_idx, d_idx in rem_changes:
                    line_count[n_idx, d_idx] -= 1
                    zero_counts[n_idx] += 1
                changes_a = try_add(a)
                if changes_a is None:
                    # roll back removal
                    for n_idx, d_idx in rem_changes:
                        line_count[n_idx, d_idx] += 1
                        zero_counts[n_idx] -= 1
                    continue
                # check full validity after both
                if not is_valid():
                    # roll back addition then removal
                    for n_idx, d_idx in changes_a:
                        line_count[n_idx, d_idx] -= 1
                        zero_counts[n_idx] += 1
                    for n_idx, d_idx in rem_changes:
                        line_count[n_idx, d_idx] += 1
                        zero_counts[n_idx] -= 1
                    continue
                # commit
                S.discard(r)
                apply_add(a, changes_a)
                outside.remove(a)
                progress = True
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
                break  # one swap per iteration; loop restarts phases
        if not progress:
            # stuck: try a fresh random restart from a different seed occasionally
            if time.time() > deadline - 120:
                break
            # small random perturbation: drop a few random points to escape local optimum
            if S:
                for _ in range(3):
                    if not S:
                        break
                    r = np.random.choice(list(S))
                    rem_changes = []
                    for d_idx in range(num_dirs):
                        for n_idx in affected[r, d_idx]:
                            if line_count[n_idx, d_idx] == 1:
                                rem_changes.append((n_idx, d_idx))
                    for n_idx, d_idx in rem_changes:
                        line_count[n_idx, d_idx] -= 1
                        zero_counts[n_idx] += 1
                    S.discard(r)
                outside = [i for i in range(NPTS) if i not in S]

    # --- Build final N = complement of best_S, dedup, mod 7 ---
    S_final = best_S
    arr = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    idx = pt_to_idx([[xa, xb], [ya, yb]])
                    if idx not in S_final:
                        arr.append([[xa % p, xb % p], [ya % p, yb % p]])
    return np.array(arr, dtype=np.int64)


# EVOLVE-BLOCK-END
