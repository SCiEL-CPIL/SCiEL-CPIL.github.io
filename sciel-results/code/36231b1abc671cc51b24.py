"""Improved randomized greedy covering with restarts and redundant-block removal."""
# EVOLVE-BLOCK-START
import time
import random
import itertools


def _greedy_cover(v, k, t, rng):
    """One randomized greedy pass. Returns a list of blocks (k-subsets)."""
    # Represent each point-set as a bitset (int) for fast membership tests.
    all_t = list(itertools.combinations(range(v), t))
    # map each t-subset to a bit index
    t_index = {c: i for i, c in enumerate(all_t)}
    n_t = len(all_t)
    # uncovered bitset as Python int bitmask
    uncovered = (1 << n_t) - 1

    # For each point, precompute the list of t-subsets it participates in (indices).
    point_subs = [[] for _ in range(v)]
    for c, i in t_index.items():
        for p in c:
            point_subs[p].append(i)

    blocks = []
    while uncovered:
        # pick a random uncovered t-subset as the target
        # find an uncovered bit: pick a random position and test
        # Efficient: collect a few candidate uncovered indices
        # We'll scan from a random offset to find an uncovered bit.
        pos = rng.randrange(n_t)
        start = pos
        while (uncovered >> start) & 1 == 0:
            start += 1
            if start >= n_t:
                start = 0
                if (uncovered >> start) & 1 == 1:
                    break
                # should not happen while uncovered != 0
                start = 0
                # full scan fallback
                while start < n_t and ((uncovered >> start) & 1) == 0:
                    start += 1
                if start >= n_t:
                    break
        if start >= n_t:
            break
        target_idx = start
        target = all_t[target_idx]

        # build block starting from target points
        block = list(target)
        block_set = set(block)

        # current "cover mask" of t-subsets this partial block already covers
        # We'll compute gain incrementally: for each candidate point x not in block,
        # gain = number of uncovered t-subsets that contain x and (t-1) points of block.
        while len(block) < k:
            # compute gains for all candidate points
            best_gain = -1
            best_points = []
            # (t-1)-subsets of current block
            block_t1 = list(itertools.combinations(block, t - 1))
            for x in range(v):
                if x in block_set:
                    continue
                # count uncovered t-subsets containing x and a (t-1)-subset of block
                g = 0
                for rest in block_t1:
                    # t-subset = rest + (x,)
                    # need its index
                    comb = tuple(sorted(rest + (x,)))
                    # find index quickly: we need a mapping from sorted t-tuple to index
                    # use a dict keyed by sorted tuple
                    idx = t_index.get(comb)
                    if idx is not None and ((uncovered >> idx) & 1):
                        g += 1
                if g > best_gain:
                    best_gain = g
                    best_points = [x]
                elif g == best_gain:
                    best_points.append(x)
            # randomized tie-break
            if best_points:
                x = best_points[rng.randrange(len(best_points))]
            else:
                # no gain; pick a random available point to fill block
                avail = [p for p in range(v) if p not in block_set]
                x = avail[rng.randrange(len(avail))] if avail else 0
            block.append(x)
            block_set.add(x)

        block.sort()
        blocks.append(block)

        # update uncovered: remove all t-subsets of this block
        for c in itertools.combinations(block, t):
            idx = t_index[c]
            uncovered &= ~(1 << idx)

    return blocks


def _remove_redundant(blocks, v, k, t):
    """Iteratively drop blocks whose t-subsets are all covered by other blocks."""
    changed = True
    while changed:
        changed = False
        # build coverage count per t-subset
        t_index = {}
        all_t = list(itertools.combinations(range(v), t))
        for c, i in zip(all_t, range(len(all_t))):
            t_index[c] = i
        n_t = len(all_t)
        # count how many blocks cover each t-subset
        count = [0] * n_t
        for b in blocks:
            seen = set()
            for c in itertools.combinations(b, t):
                idx = t_index[c]
                count[idx] += 1
        # try to drop each block if every t-subset it covers is covered by >=2 blocks
        keep = [True] * len(blocks)
        for bi, b in enumerate(blocks):
            redundant = True
            for c in itertools.combinations(b, t):
                idx = t_index[c]
                if count[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                keep[bi] = False
                # decrement counts
                for c in itertools.combinations(b, t):
                    idx = t_index[c]
                    count[idx] -= 1
                changed = True
        if changed:
            blocks = [b for b, k in zip(blocks, keep) if k]
    return blocks


def construct(v, k, t):
    deadline = time.time() + 110.0  # safety margin before 120s kill
    best = None
    best_len = None
    rng = random.Random(12345)

    # First run a deterministic-ish (seeded) greedy to get a baseline
    blocks = _greedy_cover(v, k, t, rng)
    blocks = _remove_redundant(blocks, v, k, t)
    best = blocks
    best_len = len(blocks)

    # Keep restarting with randomization until deadline
    while time.time() < deadline - 5.0:
        blocks = _greedy_cover(v, k, t, rng)
        blocks = _remove_redundant(blocks, v, k, t)
        if best_len is None or len(blocks) < best_len:
            best = blocks
            best_len = len(blocks)

    return best
# EVOLVE-BLOCK-END
