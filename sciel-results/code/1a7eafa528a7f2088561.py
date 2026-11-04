import time
import itertools
import numpy as np

# ---- Precomputation (cached per (v,k,t)) ----
_cache = {}

def _get_data(v, k, t):
    key = (v, k, t)
    if key in _cache:
        return _cache[key]

    points = list(range(v))
    # all 4-subsets, index <-> frozenset
    subs = [frozenset(c) for c in itertools.combinations(points, t)]
    sub_index = {s: i for i, s in enumerate(subs)}
    nsubs = len(subs)

    # completions[s] = array of (k - (t-1)) points that complete s (each with t-1 of s)
    completions = np.empty((nsubs, k - (t - 1)), dtype=np.int32)
    for i, s in enumerate(subs):
        sset = set(s)
        cands = [x for x in points if x not in sset]
        completions[i] = cands

    # pair_cover[p][q] = bitmask of 4-subsets containing both p and q
    pair_cover = np.zeros((v, v), dtype=object)
    for i, s in enumerate(subs):
        ss = list(s)
        for a in range(t):
            for b in range(a + 1, t):
                p, q = ss[a], ss[b]
                if pair_cover[p, q] is None:
                    pair_cover[p, q] = np.zeros(nsubs, dtype=bool)
                pair_cover[p, q][i] = True
                if p != q:
                    pair_cover[q, p] = pair_cover[p, q]

    _cache[key] = (subs, sub_index, completions, pair_cover, nsubs, points)
    return _cache[key]


def _build_greedy(data, v, k, t, seed=0):
    subs, sub_index, completions, pair_cover, nsubs, points = data
    rng = np.random.default_rng(seed)
    covered = np.zeros(nsubs, dtype=bool)
    blocks = []
    for _ in range(nsubs):
        unc = np.flatnonzero(~covered)
        if unc.size == 0:
            break
        # start block at a random uncovered subset's first element
        start = int(rng.choice(unc))
        block = [list(subs[start])[0]]
        while len(block) < k:
            block_set = set(block)
            best = -1
            best_gain = -1
            for x in points:
                if x in block_set:
                    continue
                gain = 0
                for y in block:
                    pc = pair_cover[x, y]
                    if pc is not None:
                        gain += int(np.count_nonzero(pc & ~covered))
                if gain > best_gain:
                    best_gain = gain
                    best = x
            block.append(best)
        block.sort()
        blocks.append(block)
        for c in itertools.combinations(block, t):
            covered[sub_index[frozenset(c)]] = True
    return blocks


def _cover_map(blocks, sub_index):
    m = np.zeros(len(sub_index), dtype=np.int32)
    for b in blocks:
        for c in itertools.combinations(b, 4):
            m[sub_index[frozenset(c)]] += 1
    return m


def _remove_redundant(blocks, sub_index):
    changed = True
    while changed:
        changed = False
        m = _cover_map(blocks, sub_index)
        for i, b in enumerate(blocks):
            if m[sub_index[frozenset(b)]] is not None:
                pass
            ok = True
            for c in itertools.combinations(b, 4):
                if m[sub_index[frozenset(c)]] <= 1:
                    ok = False
                    break
            if ok:
                del blocks[i]
                changed = True
                break
    return blocks


def _valid(blocks, sub_index):
    m = _cover_map(blocks, sub_index)
    return bool(np.all(m > 0))


def _local_search(blocks, data, v, k, t, deadline):
    subs, sub_index, completions, pair_cover, nsubs, points = data
    m = _cover_map(blocks, sub_index)
    best = list(blocks)
    best_n = len(blocks)
    cur = list(blocks)
    rng = np.random.default_rng(12345)
    while time.time() < deadline:
        # --- remove a redundant block ---
        removed = -1
        for i, b in enumerate(cur):
            for c in itertools.combinations(b, 4):
                if m[sub_index[frozenset(c)]] <= 1:
                    break
            else:
                removed = i
                break
        if removed >= 0:
            b = cur.pop(removed)
            for c in itertools.combinations(b, 4):
                m[sub_index[frozenset(c)]] -= 1
            if len(cur) < best_n:
                best_n = len(cur)
                best = list(cur)
            continue
        # --- swap: remove one, add one that covers most currently-uncovered ---
        if len(cur) >= best_n:
            i = int(rng.integers(0, len(cur)))
            b = cur[i]
            for c in itertools.combinations(b, 4):
                m[sub_index[frozenset(c)]] -= 1
            unc = np.flatnonzero(m == 0)
            if unc.size > 0:
                # build a greedy block targeting uncovered subsets
                start = int(rng.choice(unc))
                nb = [list(subs[start])[0]]
                while len(nb) < k:
                    nb_set = set(nb)
                    bestx, bestg = -1, -1
                    for x in points:
                        if x in nb_set:
                            continue
                        g = 0
                        for y in nb:
                            pc = pair_cover[x, y]
                            if pc is not None:
                                g += int(np.count_nonzero(pc & (m == 0)))
                        if g > bestg:
                            bestg = g
                            bestx = x
                    nb.append(bestx)
                nb.sort()
                cur[i] = nb
                for c in itertools.combinations(nb, 4):
                    m[sub_index[frozenset(c)]] += 1
                if _valid(cur, sub_index) and len(cur) < best_n:
                    best_n = len(cur)
                    best = list(cur)
                # if not better, keep it anyway (it's valid or not; revert if invalid)
                if not _valid(cur, sub_index):
                    # revert swap
                    for c in itertools.combinations(nb, 4):
                        m[sub_index[frozenset(c)]] -= 1
                    for c in itertools.combinations(b, 4):
                        m[sub_index[frozenset(c)]] += 1
                    cur[i] = b
            else:
                # no uncovered -> revert (swap didn't help)
                for c in itertools.combinations(b, 4):
                    m[sub_index[frozenset(c)]] += 1
                cur[i] = b
    return best


def construct(v, k, t):
    start = time.time()
    deadline = start + 115.0
    data = _get_data(v, k, t)
    subs, sub_index, completions, pair_cover, nsubs, points = data

    # initial greedy covering
    blocks = _build_greedy(data, v, k, t, seed=0)
    # trim redundancy
    blocks = _remove_redundant(blocks, sub_index)

    best = list(blocks)
    best_n = len(blocks)

    # local search within deadline
    ls_end = deadline - 5.0
    while time.time() < ls_end:
        cur = _local_search(best, data, v, k, t, min(time.time() + 8.0, ls_end))
        if len(cur) < best_n:
            best_n = len(cur)
            best = cur
            blocks = _remove_redundant(best, sub_index)
            if len(blocks) < best_n:
                best_n = len(blocks)
                best = blocks

    # safety: ensure validity
    if not _valid(best, sub_index):
        # fall back to a fresh greedy
        fb = _build_greedy(data, v, k, t, seed=1)
        fb = _remove_redundant(fb, sub_index)
        if len(fb) < len(best):
            best = fb

    return best
