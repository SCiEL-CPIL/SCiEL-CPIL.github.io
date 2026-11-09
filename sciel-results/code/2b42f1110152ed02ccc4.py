"""Improved randomized greedy covering with restarts and redundant-block removal."""
# EVOLVE-BLOCK-START
import time
import random
import itertools


def _greedy_cover(v, k, t, rng, t_index, all_t):
    """One randomized greedy pass. Returns a list of blocks (k-subsets)."""
    n_t = len(all_t)
    uncovered = (1 << n_t) - 1
    blocks = []
    while uncovered:
        # pick a random uncovered t-subset as the target
        target_idx = None
        for _ in range(n_t):
            cand = rng.randrange(n_t)
            if (uncovered >> cand) & 1:
                target_idx = cand
                break
        if target_idx is None:
            break
        target = all_t[target_idx]

        block = list(target)
        block_set = set(block)

        while len(block) < k:
            block_t1 = list(itertools.combinations(block, t - 1))
            best_gain = -1
            best_points = []
            for x in range(v):
                if x in block_set:
                    continue
                g = 0
                for rest in block_t1:
                    comb = tuple(sorted(rest + (x,)))
                    idx = t_index.get(comb)
                    if idx is not None and ((uncovered >> idx) & 1):
                        g += 1
                if g > best_gain:
                    best_gain = g
                    best_points = [x]
                elif g == best_gain:
                    best_points.append(x)
            if best_points:
                x = best_points[rng.randrange(len(best_points))]
            else:
                avail = [p for p in range(v) if p not in block_set]
                x = avail[rng.randrange(len(avail))] if avail else 0
            block.append(x)
            block_set.add(x)

        block.sort()
        blocks.append(block)

        for c in itertools.combinations(block, t):
            idx = t_index[c]
            uncovered &= ~(1 << idx)

    return blocks


def _remove_redundant(blocks, v, k, t, t_index):
    """Iteratively drop blocks whose t-subsets are all covered by other blocks."""
    while True:
        n_t = len(t_index)
        count = [0] * n_t
        block_subs = []
        for b in blocks:
            subs = [t_index[c] for c in itertools.combinations(b, t)]
            block_subs.append(subs)
            for idx in subs:
                count[idx] += 1
        changed = False
        keep = [True] * len(blocks)
        for bi, subs in enumerate(block_subs):
            redundant = all(count[idx] >= 2 for idx in subs)
            if redundant:
                keep[bi] = False
                for idx in subs:
                    count[idx] -= 1
                changed = True
        if not changed:
            break
        blocks = [b for b, k in zip(blocks, keep) if k]
    return blocks


def construct(v, k, t):
    deadline = time.time() + 140.0  # safety margin before 150s kill
    rng = random.Random(12345)

    all_t = list(itertools.combinations(range(v), t))
    t_index = {c: i for i, c in enumerate(all_t)}

    best = None
    best_len = None

    blocks = _greedy_cover(v, k, t, rng, t_index, all_t)
    blocks = _remove_redundant(blocks, v, k, t, t_index)
    best = blocks
    best_len = len(blocks)

    while time.time() < deadline - 5.0:
        blocks = _greedy_cover(v, k, t, rng, t_index, all_t)
        blocks = _remove_redundant(blocks, v, k, t, t_index)
        if best_len is None or len(blocks) < best_len:
            best = blocks
            best_len = len(blocks)

    return best
# EVOLVE-BLOCK-END
