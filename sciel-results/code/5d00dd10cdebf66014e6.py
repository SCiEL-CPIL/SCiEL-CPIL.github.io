# EVOLVE-BLOCK-START
import time
import numpy as np


def _shannon(p):
    p = p[p > 0]
    return float(np.sum(-p * np.log2(p)))


def _proj_entropy(P, r):
    # P[y, x]; value = x + r*y
    n = P.shape[0]
    vals = np.zeros((n, n))
    for y in range(n):
        for x in range(n):
            vals[y, x] = x + r * y
    # pool by value
    v, idx = np.unique(vals, return_inverse=True)
    sums = np.bincount(idx.ravel(), weights=P.ravel())
    return _shannon(sums)


def _score(P):
    D = _proj_entropy(P, -1.0)
    H0 = _proj_entropy(P, 0.0)
    H1 = _proj_entropy(P, 1.0)
    H2 = _proj_entropy(P, 2.0)
    M = max(H0, H1, H2)
    if M <= 1e-9:
        return 0.0
    return D / M


def arithmetic_kakeya_distribution() -> np.ndarray:
    start = time.time()
    DEADLINE = start + 620.0  # safety margin before 630s kill
    n = 10

    # Candidate cells: each (y, x). We'll search over subsets with a fixed total mass,
    # assigning equal weight to selected cells.
    # Represent a support by a boolean mask over the 100 cells.
    all_cells = [(y, x) for y in range(n) for x in range(n)]
    idx_of = {(y, x): i for i, (y, x) in enumerate(all_cells)}

    # Start from a reasonable seed: a sparse diagonal-like support.
    # We'll do a greedy local search over supports of varying size.

    def mask_to_P(mask):
        P = np.zeros((n, n), dtype=np.float64)
        cnt = int(np.sum(mask))
        if cnt == 0:
            return P
        for (y, x) in all_cells:
            if mask[idx_of[(y, x)]]:
                P[y, x] = 1.0 / cnt
        return P

    best_P = np.ones((n, n), dtype=np.float64) / (n * n)
    best_score = _score(best_P)

    # Precompute per-cell projection values for fast recompute
    # For a support mask with equal weights, projection entropy depends only on counts per value.
    # value for slope r at cell (y,x) = x + r*y
    val_neg1 = np.zeros((n, n), dtype=int)  # x - y
    val_0 = np.zeros((n, n), dtype=int)     # x
    val_1 = np.zeros((n, n), dtype=int)     # x + y
    val_2 = np.zeros((n, n), dtype=int)     # x + 2y
    for y in range(n):
        for x in range(n):
            val_neg1[y, x] = x - y
            val_0[y, x] = x
            val_1[y, x] = x + y
            val_2[y, x] = x + 2 * y

    def entropy_from_counts(counts, total):
        # counts: dict or array of counts per value; total = sum
        probs = np.array([c / total for c in counts if c > 0], dtype=np.float64)
        if len(probs) == 0:
            return 0.0
        return float(np.sum(-probs * np.log2(probs)))

    def score_from_mask(mask):
        total = int(np.sum(mask))
        if total == 0:
            return 0.0
        # count per value for each slope
        cn_neg1 = {}
        cn_0 = {}
        cn_1 = {}
        cn_2 = {}
        for (y, x) in all_cells:
            i = idx_of[(y, x)]
            if mask[i]:
                cn_neg1[val_neg1[y, x]] = cn_neg1.get(val_neg1[y, x], 0) + 1
                cn_0[val_0[y, x]] = cn_0.get(val_0[y, x], 0) + 1
                cn_1[val_1[y, x]] = cn_1.get(val_1[y, x], 0) + 1
                cn_2[val_2[y, x]] = cn_2.get(val_2[y, x], 0) + 1
        D = entropy_from_counts(list(cn_neg1.values()), total)
        H0 = entropy_from_counts(list(cn_0.values()), total)
        H1 = entropy_from_counts(list(cn_1.values()), total)
        H2 = entropy_from_counts(list(cn_2.values()), total)
        M = max(H0, H1, H2)
        if M <= 1e-9:
            return 0.0
        return D / M

    # Greedy construction: start empty, add cell that most increases score.
    # Then do local search: flip individual cells (add/remove) and swaps.

    def greedy_construct(target_size):
        mask = np.zeros(n * n, dtype=bool)
        for _ in range(target_size):
            best_gain = -1e9
            best_cell = None
            for (y, x) in all_cells:
                i = idx_of[(y, x)]
                if mask[i]:
                    continue
                mask[i] = True
                s = score_from_mask(mask)
                mask[i] = False
                if s > best_gain:
                    best_gain = s
                    best_cell = i
            if best_cell is None:
                break
            mask[best_cell] = True
        return mask

    def local_search(mask, iters):
        cur = mask.copy()
        cur_score = score_from_mask(cur)
        improved = True
        step = 0
        while improved and step < iters:
            improved = False
            step += 1
            if time.time() > DEADLINE - 60:
                break
            best_score_local = cur_score
            best_mask = cur
            # try adding a cell
            for (y, x) in all_cells:
                i = idx_of[(y, x)]
                if cur[i]:
                    continue
                cur[i] = True
                s = score_from_mask(cur)
                cur[i] = False
                if s > best_score_local + 1e-12:
                    best_score_local = s
                    best_mask = cur.copy()
                    best_mask[i] = True
            # try removing a cell
            for (y, x) in all_cells:
                i = idx_of[(y, x)]
                if not cur[i]:
                    continue
                cur[i] = False
                s = score_from_mask(cur)
                cur[i] = True
                if s > best_score_local + 1e-12:
                    best_score_local = s
                    best_mask = cur.copy()
                    best_mask[i] = False
            if best_mask is not cur:
                cur = best_mask
                cur_score = best_score_local
                improved = True
        return cur, cur_score

    # Try several support sizes via greedy + local search
    for size in [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]:
        if time.time() > DEADLINE - 120:
            break
        mask = greedy_construct(size)
        mask, s = local_search(mask, iters=40)
        P = mask_to_P(mask)
        s2 = _score(P)
        if s2 > best_score:
            best_score = s2
            best_P = P

    # Also try a random-restart local search from random sparse masks
    rng = np.random.default_rng(0)
    for _ in range(200):
        if time.time() > DEADLINE - 60:
            break
        size = int(rng.integers(3, 16))
        mask = np.zeros(n * n, dtype=bool)
        chosen = rng.choice(n * n, size=size, replace=False)
        mask[chosen] = True
        mask, s = local_search(mask, iters=25)
        P = mask_to_P(mask)
        s2 = _score(P)
        if s2 > best_score:
            best_score = s2
            best_P = P

    # Final safety: ensure valid
    best_P = np.clip(best_P, 0.0, None)
    tot = best_P.sum()
    if tot <= 1e-9:
        best_P = np.ones((n, n), dtype=np.float64) / (n * n)
    else:
        best_P = best_P / tot

    return best_P


# EVOLVE-BLOCK-END
