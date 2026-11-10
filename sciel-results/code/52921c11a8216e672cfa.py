# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def _comb_index(subset, v, t):
    """Combinatorial number system: index of a t-subset (ascending) among all C(v,t)."""
    # subset is a tuple/list of t ascending ints in 0..v-1
    idx = 0
    for i in range(t):
        idx += _binom(subset[i], i + 1)
    return idx


def _precompute_binom(v, t):
    # binom[n][r] = C(n, r) for 0<=n<=v, 0<=r<=t
    b = np.zeros((v + 1, t + 1), dtype=np.int64)
    for n in range(v + 1):
        b[n][0] = 1
        for r in range(1, t + 1):
            if r <= n:
                b[n][r] = b[n - 1][r - 1] + b[n - 1][r]
    return b


_BINOM_CACHE = {}


def _binom(n, r):
    key = (n, r)
    cache = _BINOM_CACHE.get(r)
    if cache is None:
        # build for this r up to needed n; lazy
        cache = {}
        _BINOM_CACHE[r] = cache
    if n in cache:
        return cache[n]
    # compute C(n,r) iteratively
    if r < 0 or r > n:
        val = 0
    else:
        r = min(r, n - r)
        val = 1
        for i in range(1, r + 1):
            val = val * (n - i + 1) // i
    cache[n] = val
    return val


def _tsubsets_of_block(block, v, t):
    """Yield _comb_index for each t-subset of block (block is a sorted list of ints)."""
    return (_comb_index(s, v, t) for s in itertools.combinations(block, t))


def _build_tsubsets_index_map(v, t):
    """Return (all_indices, idx_of_subset) where all_indices is array of length C(v,t)."""
    total = _binom(v, t)
    all_indices = np.arange(total, dtype=np.int64)
    # map from tuple -> index
    idx_map = {}
    for i, s in enumerate(itertools.combinations(range(v), t)):
        idx_map[s] = i
    return all_indices, idx_map


def construct(v, k, t):
    start = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    # Precompute binom table for fast _comb_index
    b = _precompute_binom(v, t)

    def comb_idx(subset):
        idx = 0
        for i in range(t):
            idx += int(b[subset[i]][i + 1])
        return idx

    # All t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    total_t = len(all_tsubs)

    # coverage count per t-subset (how many blocks cover it)
    cover = np.zeros(total_t, dtype=np.int32)
    uncovered = np.ones(total_t, dtype=bool)
    remaining = total_t

    blocks = []  # list of frozensets (or tuples)
    # Precompute, for each candidate point, the list of t-subsets it participates in?
    # Instead, for greedy, compute gain by enumerating t-subsets of partial block + new point.

    # For speed, precompute for each t-subset its points as a list (for coverage updates)
    # and a mapping from (t-1 subset) -> list of t-subsets containing it (to compute gains fast)
    # Build: for each (t-1)-subset, list of t-subset indices that contain it
    t1_map = {}
    for idx, s in enumerate(all_tsubs):
        for j in range(t):
            t1 = s[:j] + s[j + 1:]
            t1_map.setdefault(t1, []).append(idx)

    def gain_of_point(x, block_pts):
        # block_pts is current list of points in the block (length < k)
        # gain = number of currently-uncovered t-subsets that contain x and (t-1) of block_pts
        g = 0
        for rest in itertools.combinations(block_pts, t - 1):
            lst = t1_map.get(rest)
            if lst:
                g += int(np.count_nonzero(uncovered[lst]))
        return g

    def add_block(block_list):
        nonlocal remaining
        block_list = sorted(block_list)
        # update cover counts
        for s in itertools.combinations(block_list, t):
            idx = comb_idx(s)
            if cover[idx] == 0:
                remaining -= 1
                uncovered[idx] = False
            cover[idx] += 1
        blocks.append(tuple(block_list))

    # Greedy construction
    while remaining > 0:
        if time.time() > DEADLINE:
            break
        # Pick a block: start from an uncovered t-subset, grow it
        # Find an uncovered t-subset index
        unc_idx = int(np.argmax(uncovered))
        if not uncovered[unc_idx]:
            break
        # initial block from that t-subset
        block = list(all_tsubs[unc_idx])
        # grow to k points choosing max gain
        while len(block) < k:
            best_x = -1
            best_g = -1
            for x in range(v):
                if x in block:
                    continue
                g = gain_of_point(x, block)
                if g > best_g:
                    best_g = g
                    best_x = x
            if best_x < 0:
                break
            block.append(best_x)
        add_block(block)

    # If we broke out early due to deadline, we may not have a full cover.
    # Ensure we still return something; but we need validity. So if not fully covered,
    # we must finish covering. But deadline... We'll do a final fast greedy to finish.
    if remaining > 0:
        # finish with a simple loop (may exceed deadline but must be valid)
        while remaining > 0:
            unc_idx = int(np.argmax(uncovered))
            if not uncovered[unc_idx]:
                break
            block = list(all_tsubs[unc_idx])
            while len(block) < k:
                best_x = -1
                best_g = -1
                for x in range(v):
                    if x in block:
                        continue
                    g = gain_of_point(x, block)
                    if g > best_g:
                        best_g = g
                        best_x = x
                if best_x < 0:
                    break
                block.append(best_x)
            add_block(block)

    # Redundancy removal: try to remove blocks that are not needed
    def is_removable(i):
        # simulate removing block i: check if any t-subset it covers would drop to 0
        bl = blocks[i]
        for s in itertools.combinations(bl, t):
            idx = comb_idx(s)
            if cover[idx] == 1:
                return False
        return True

    # Iteratively remove removable blocks (pass until no change)
    changed = True
    while changed and (time.time() < DEADLINE - 5.0):
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            if is_removable(i):
                bl = blocks[i]
                for s in itertools.combinations(bl, t):
                    idx = comb_idx(s)
                    cover[idx] -= 1
                    if cover[idx] == 0:
                        uncovered[idx] = True
                        remaining += 1
                blocks.pop(i)
                changed = True

    # Return as list of lists
    return [list(b) for b in blocks]


# EVOLVE-BLOCK-END
