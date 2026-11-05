# EVOLVE-BLOCK-START
import time
import itertools
import random


def construct(v, k, t):
    deadline = time.time() + 130.0  # safety margin under 150s wall-clock
    tsubsets = list(itertools.combinations(range(v), t))
    n_t = len(tsubsets)
    idx = {s: i for i, s in enumerate(tsubsets)}

    # Precompute per-point bitmask: bit i set iff point x lies in t-subset i
    point_mask = [0] * v
    for i, s in enumerate(tsubsets):
        bit = 1 << i
        for x in s:
            point_mask[x] |= bit

    def gain(mask, block, block_mask):
        # uncovered subsets that contain all current block points and point x
        m = mask & block_mask & point_mask[x]
        return m.bit_count()

    def greedy(seed):
        rng = random.Random(seed)
        uncovered = (1 << n_t) - 1
        blocks = []
        while uncovered:
            # pick an uncovered t-subset (lowest set bit)
            lsb = uncovered & (-uncovered)
            i = lsb.bit_length() - 1
            block = list(tsubsets[i])
            block_mask = 0
            for x in block:
                block_mask |= point_mask[x]
            while len(block) < k:
                best_x = -1
                best_g = -1
                for x in range(v):
                    if block_mask & point_mask[x]:
                        continue  # x already in block
                    g = gain(uncovered, block, block_mask)
                    if g > best_g:
                        best_g = g
                        best_x = x
                if best_x < 0:
                    # should not happen; fill arbitrarily
                    for x in range(v):
                        if not (block_mask & point_mask[x]):
                            best_x = x
                            break
                block.append(best_x)
                block_mask |= point_mask[best_x]
            blocks.append(block)
            uncovered &= ~block_mask
        return blocks

    # deterministic baseline first (also gives an early valid answer)
    best = greedy(0)
    best_len = len(best)

    # randomized restarts until deadline
    s = 1
    while time.time() < deadline - 2.0:
        cand = greedy(s)
        if len(cand) < best_len:
            best = cand
            best_len = len(cand)
        s += 1

    return best
# EVOLVE-BLOCK-END
