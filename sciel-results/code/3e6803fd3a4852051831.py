"""Additive bases with few representations: minimize peak pair-sum multiplicity."""
import time
import random
import numpy as np

_N = 128
_LO, _HI = 128, 256
_DEADLINE = 80.0  # safety margin under 90s wall-clock


def _pair_sum_counts(A_mask):
    """Return array cnt[s] for s in [128,256] of unordered pair counts (a<=b, a+b=s)."""
    A = np.nonzero(A_mask)[0]
    m = len(A)
    cnt = np.zeros(_HI + 1, dtype=np.int64)
    for i in range(m):
        ai = A[i]
        # b >= ai, b in A, ai+b in [128,256]
        # b >= 128 - ai, b <= 256 - ai
        lo = max(ai, 128 - ai)
        hi = min(128, 256 - ai)
        if lo > hi:
            continue
        idx = np.searchsorted(A, lo)
        end = np.searchsorted(A, hi + 1)
        Bs = A[idx:end]
        sums = ai + Bs
        # only those in [128,256]
        ok = (sums >= 128) & (sums <= 256)
        if ok.any():
            np.add.at(cnt, sums[ok], 1)
    return cnt[128:257]


def construct():
    start = time.time()
    rng = random.Random(12345)
    np_rng = np.random.default_rng(12345)

    # baseline: the seed
    base_mask = np.zeros(_N + 1, dtype=bool)
    base_mask[64:129] = True
    base_cnt = _pair_sum_counts(base_mask)
    base_peak = int(base_cnt.max())
    best_A = list(np.nonzero(base_mask)[0])
    best_peak = base_peak

    def elapsed():
        return time.time() - start

    def peak_of(cnt):
        return int(cnt.max())

    def is_valid(cnt):
        # every target must be covered
        return bool((cnt >= 1).all())

    # We'll run local search minimizing (peak, then sum of squares) with coverage hard.
    # Objective: primary = peak multiplicity; we want it as low as possible while covering all.
    def objective(cnt):
        if not (cnt >= 1).all():
            missing = int((cnt == 0).sum())
            # heavy penalty for missing
            return (10**6 + missing, int(cnt.max()))
        return (0, int(cnt.max()))

    cur_mask = base_mask.copy()
    cur_cnt = _pair_sum_counts(cur_mask)
    cur_obj = objective(cur_cnt)

    # candidate pool: all elements 0..128
    all_elems = list(range(_N + 1))

    def toggle(i, add):
        nonlocal cur_mask, cur_cnt, cur_obj
        if add:
            cur_mask[i] = True
        else:
            cur_mask[i] = False
        cur_cnt = _pair_sum_counts(cur_mask)
        cur_obj = objective(cur_cnt)

    # Simulated-annealing-ish local search with restarts on K target.
    # Strategy: try to achieve peak <= K for decreasing K.
    # Start from a good random init that covers everything.

    def random_init():
        # start from base, then randomly add some low elements and remove some high to diversify
        m = base_mask.copy()
        # randomly toggle a few
        for _ in range(20):
            e = rng.choice(all_elems)
            m[e] = not m[e]
        # ensure coverage; if not, add back base
        cnt = _pair_sum_counts(m)
        if not (cnt >= 1).all():
            m = base_mask.copy()
        return m

    best_overall = best_peak

    # Try a sequence of target peaks from low to high; for each, run local search trying to
    # get a valid set with peak <= target. If found, record and try lower.
    # But we don't know the true minimum; just minimize directly via local search.

    iters = 0
    while elapsed() < _DEADLINE:
        # local search: random single-bit flips accepting if objective improves or equal
        for _ in range(2000):
            if elapsed() > _DEADLINE:
                break
            iters += 1
            e = rng.choice(all_elems)
            # decide add/remove
            if cur_mask[e]:
                add = False
            else:
                add = True
            old_obj = cur_obj
            toggle(e, add)
            if cur_obj <= old_obj:
                # accept
                if cur_obj[0] == 0:
                    if cur_obj[1] < best_overall:
                        best_overall = cur_obj[1]
                        best_A = list(np.nonzero(cur_mask)[0])
                else:
                    # invalid; keep if still improving on missing count? we only accept <=, so fine
                    pass
            else:
                # reject
                toggle(e, not add)
        # occasional restart from a fresh random init if stuck
        if iters % 20000 == 0:
            nm = random_init()
            nc = _pair_sum_counts(nm)
            no = objective(nc)
            if no < cur_obj:
                cur_mask, cur_cnt, cur_obj = nm, nc, no

    # Final: return best valid set found (min peak). If none valid, return base.
    # Verify best_A validity
    bm = np.zeros(_N + 1, dtype=bool)
    for e in best_A:
        bm[e] = True
    bc = _pair_sum_counts(bm)
    if (bc >= 1).all():
        return best_A
    return best_A  # base is always valid


# quick self-test (not run by evaluator)
if __name__ == "__main__":
    t0 = time.time()
    A = construct()
    print("time", time.time() - t0)
    print("len A", len(A))
    cnt = _pair_sum_counts(np.array([1 if e in set(A) else 0 for e in range(_N+1)], dtype=bool))
    print("peak", cnt.max(), "min", cnt.min())
