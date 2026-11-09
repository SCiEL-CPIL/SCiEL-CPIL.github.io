# EVOLVE-BLOCK-START
import time
import numpy as np

_N = 192
_K = 3
_MAX = 48


def _is_valid(sel):
    """True iff no selected integer divides the sum of any k=3 distinct OTHER selected integers."""
    s = np.asarray(sel, dtype=np.int64)
    n = s.size
    if n < _K + 1:
        return True
    idx = np.arange(n)
    a = s[:, None]
    b = s[None, :]
    mask = (a != b) & (idx[:, None] != idx[None, :])
    ab = a * b
    ab = np.where(mask, ab, 0)
    ab_sum = ab.sum(axis=0)  # per j: sum of s_i over i != j
    # pair sums over distinct i<j
    ii, jj = np.triu_indices(n, 1)
    psum = s[ii] + s[jj]
    # per j: sum over pairs {i1,i2} distinct, both != j of s_i1+s_i2
    # = 2*(ab_sum_j - s_j) - (sum over i1<i2, i1!=j, i2!=j of s_i1 + s_i2)
    # The last double sum = for each i1 (i1!=j): s_i1*(# of i2 with i2<i1, i2!=j) + s_i1*(# i2>i1, i2!=j)
    # = sum_{i1!=j} s_i1 * (n - 1 - 1) ... = (n-2)*ab_sum_j  (each i1 counted once per other, total n-2 partners)
    # So pairsum_j = 2*(ab_sum_j - s_j) - (n-2)*ab_sum_j = (4-n)*ab_sum_j - 2*s_j
    pairsum = (4 - n) * ab_sum - 2 * s
    triple = pairsum[:, None] - s[None, :]  # triple[a,b] = sum of 3 distinct others for a, excluding b
    # violation for (a, b): a divides triple[a,b]
    viol = (triple % a) == 0
    # exclude b == a (not a distinct other)
    viol = viol & (idx[:, None] != idx[None, :])
    return not viol.any()


def _score(sel):
    """Number of (a,b) divisibility violations (0 means fully valid)."""
    s = np.asarray(sel, dtype=np.int64)
    n = s.size
    if n < _K + 1:
        return 0
    idx = np.arange(n)
    a = s[:, None]
    b = s[None, :]
    mask = (a != b) & (idx[:, None] != idx[None, :])
    ab = a * b
    ab = np.where(mask, ab, 0)
    ab_sum = ab.sum(axis=0)
    ii, jj = np.triu_indices(n, 1)
    psum = s[ii] + s[jj]
    pairsum = (4 - n) * ab_sum - 2 * s
    triple = pairsum[:, None] - s[None, :]
    viol = (triple % a) == 0
    viol = viol & (idx[:, None] != idx[None, :])
    return int(viol.sum())


def _greedy_seed():
    """Greedy descending build: add x if it keeps the set valid. Fast safety floor."""
    sel = []
    for x in range(_N, 0, -1):
        cand = sel + [x]
        if _is_valid(cand):
            sel.append(x)
    return sel


def construct():
    deadline = time.time() + 52.0  # safety margin vs 60s
    best = _greedy_seed()
    best_score = _score(best)  # 0 if valid
    best_val = len(best)

    rng = np.random.default_rng(20240517)
    universe = np.arange(1, _N + 1, dtype=np.int64)

    it = 0
    while time.time() < deadline:
        it += 1
        # start from a random 48-subset
        cur = universe[rng.choice(_N, size=_MAX, replace=False)]
        cur_score = _score(cur)
        # local search: try to reduce violations, then grow toward validity
        improved = True
        while improved and time.time() < deadline:
            improved = False
            cur_list = cur.tolist()
            # collect candidate moves
            # 1-1 swap: replace one cur element with one outside
            cur_set = set(cur_list)
            outside = [v for v in range(1, _N + 1) if v not in cur_set]
            if not outside:
                break
            # sample a bounded number of moves to keep it fast
            n_swaps = min(400, len(cur_list) * len(outside))
            for _ in range(n_swaps):
                i = int(rng.integers(0, len(cur_list)))
                j = int(rng.integers(0, len(outside)))
                new_list = cur_list[:]
                new_list[i] = outside[j]
                ns = _score(new_list)
                if ns < cur_score:
                    cur_score = ns
                    cur_list = new_list
                    cur_set = set(cur_list)
                    outside = [v for v in range(1, _N + 1) if v not in cur_set]
                    improved = True
                    break
        # try to grow: add elements while staying valid (or reducing score)
        # If currently valid (score 0), we are done growing this trial.
        if cur_score == 0:
            # try to add more elements greedily to increase cardinality
            cur_list = cur.tolist()
            cur_set = set(cur_list)
            for v in range(1, _N + 1):
                if v in cur_set:
                    continue
                cand = cur_list + [v]
                if _is_valid(cand):
                    cur_list = cand
                    cur_set.add(v)
            cur_score = _score(cur_list)
            if len(cur_list) > best_val:
                best_val = len(cur_list)
                best = cur_list
                best_score = cur_score
            if best_val >= _MAX and best_score == 0:
                break
        else:
            # track best partial if it beats current best by validity then cardinality
            if cur_score < best_score or (cur_score == best_score and len(cur) > best_val):
                best_score = cur_score
                best_val = len(cur)
                best = cur.tolist()

        if best_val >= _MAX and best_score == 0:
            break

    # Final: ensure best is valid if possible; if best has violations, fall back to a valid subset
    if best_score != 0:
        # build a valid set from best by dropping offending elements
        # simple: recompute a valid subset of best via greedy over best elements
        valid = []
        for v in best:
            cand = valid + [v]
            if _is_valid(cand):
                valid.append(v)
        # also try to add from full universe to grow
        valid_set = set(valid)
        for v in range(1, _N + 1):
            if v in valid_set:
                continue
            cand = valid + [v]
            if _is_valid(cand):
                valid.append(v)
                valid_set.add(v)
        if len(valid) >= len(best):
            best = valid
            best_val = len(valid)
            best_score = _score(best)

    return best
# EVOLVE-BLOCK-END
