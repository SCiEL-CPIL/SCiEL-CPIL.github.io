"""Greedy covering C(v,k,t): numpy bitset + per-point counts + budgeted randomized restarts + repair."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np


def _comb(n, r):
    if r < 0 or r > n:
        return 0
    r = min(r, n - r)
    res = 1
    for i in range(r):
        res = res * (n - i) // (i + 1)
    return res


def _gid(sub, v):
    g = 0
    for x in sub:
        g = g * v + x
    return g


def construct(v, k, t):
    deadline = time.time() + 140.0
    Cvt = _comb(v, t)
    nwords = (Cvt + 63) // 64
    W = np.int64

    # Precompute all t-subsets (gids) and, for each, the set of points it contains.
    tsubs = list(itertools.combinations(range(v), t))
    tsub_gids = np.array([_gid(s, v) for s in tsubs], dtype=np.int64)
    tsub_word = tsub_gids >> 6
    tsub_bit = tsub_gids & 63

    # For each point p, which t-subsets contain it (list of tsub indices)
    pts_to_tsubs = [[] for _ in range(v)]
    for i, s in enumerate(tsubs):
        for p in s:
            pts_to_tsubs[p].append(i)

    # Precompute, for each k-subset, the set of t-subset indices it covers.
    ksubs = list(itertools.combinations(range(v), k))
    # cover_mask[kidx] : boolean array length Cvt of covered tsub indices
    cover_mask = np.zeros((len(ksubs), Cvt), dtype=bool)
    for ki, combo in enumerate(ksubs):
        for sub in itertools.combinations(combo, t):
            cover_mask[ki, _gid(sub, v)] = True

    def block_gains(block, covered):
        # returns array length v of number of still-uncovered tsubsets each candidate point would add
        # a point p adds a t-subset s iff s contains p and s is uncovered and s\\{p} subset of block
        gains = np.zeros(v, dtype=np.int64)
        # t-subsets that are subsets of block (already covered by block so far)
        # For each uncovered t-subset s, the points it still needs = s - block
        # A candidate p covers s iff p in (s - block) and (s - {p}) subset block
        for i, s in enumerate(tsubs):
            if covered[i]:
                continue
            # does s intersect block in >= t-1 points?
            inter = sum(1 for x in s if x in block)
            if inter >= t - 1:
                # points in s not in block can each complete s
                for p in s:
                    if p not in block:
                        gains[p] += 1
        return gains

    def run_greedy(start_pts, start_time):
        covered = np.zeros(Cvt, dtype=bool)
        blocks = []
        block = list(start_pts)
        used = set(block)
        while len(block) < k:
            g = block_gains(block, covered)
            for p in used:
                g[p] = -1
            best_p = int(np.argmax(g))
            if g[best_p] <= 0:
                # pick any unused
                for p in range(v):
                    if p not in used:
                        best_p = p
                        break
            block.append(best_p)
            used.add(best_p)
        block.sort()
        blocks.append(block)
        # mark covered
        for sub in itertools.combinations(block, t):
            covered[_gid(sub, v)] = True

        while True:
            if time.time() > start_time + 5.0:
                return None
            if covered.all():
                break
            # start new block at point with max uncovered tsubsets containing it
            cnt = np.zeros(v, dtype=np.int64)
            for i in range(Cvt):
                if not covered[i]:
                    for p in tsubs[i]:
                        cnt[p] += 1
            start = int(np.argmax(cnt))
            block = [start]
            used = {start}
            while len(block) < k:
                g = block_gains(block, covered)
                for p in used:
                    g[p] = -1
                best_p = int(np.argmax(g))
                if g[best_p] <= 0:
                    for p in range(v):
                        if p not in used:
                            best_p = p
                            break
                block.append(best_p)
                used.add(best_p)
            block.sort()
            blocks.append(block)
            for sub in itertools.combinations(block, t):
                covered[_gid(sub, v)] = True
        return blocks

    best = None
    best_n = None

    r = run_greedy(list(range(min(k, v))), time.time())
    if r:
        best, best_n = r, len(r)

    rng = np.random.default_rng(0)
    while time.time() < deadline - 12.0:
        seed = list(rng.choice(v, size=min(k, v), replace=False))
        r = run_greedy(seed, time.time())
        if r and (best_n is None or len(r) < best_n):
            best, best_n = r, len(r)
            if best_n <= 131:
                break

    if best is None:
        best = []
    return best
# EVOLVE-BLOCK-END
