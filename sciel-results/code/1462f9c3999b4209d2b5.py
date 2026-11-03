"""Greedy set cover for C(19,8,5) with dual weights and incremental counts."""
import time
import random
import itertools


def construct(v, k, t):
    start = time.time()
    deadline = start + 112.0  # safety margin before 120s kill

    points = range(v)
    tsubs = list(itertools.combinations(points, t))
    T = len(tsubs)

    # For each point, list of t-subset indices containing it
    point_tsubs = [[] for _ in range(v)]
    for idx, ts in enumerate(tsubs):
        for x in ts:
            point_tsubs[x].append(idx)

    # All k-blocks and the t-subsets each covers
    kblocks = list(itertools.combinations(points, k))
    B = len(kblocks)
    block_tsubs = []
    for blk in kblocks:
        s = set()
        for x in blk:
            s.update(point_tsubs[x])
        block_tsubs.append(sorted(s))

    # Reverse: for each t-subset, the list of blocks containing it
    tsub_blocks = [[] for _ in range(T)]
    for b, lst in enumerate(block_tsubs):
        for i in lst:
            tsub_blocks[i].append(b)

    best = None
    best_len = None

    def greedy(weights_init, decay):
        weights = weights_init[:]
        covered = [False] * T
        block_score = [0.0] * B
        for b, lst in enumerate(block_tsubs):
            s = 0.0
            for i in lst:
                s += weights[i]
            block_score[b] = s
        blocks = []
        chosen_set = set()
        while True:
            best_b = -1
            best_score = -1.0
            for b, sc in enumerate(block_score):
                if sc > best_score:
                    best_score = sc
                    best_b = b
            if best_b < 0 or best_score <= 0:
                forced = False
                for i in range(T):
                    if not covered[i]:
                        for b2 in tsub_blocks[i]:
                            if b2 not in chosen_set:
                                best_b = b2
                                forced = True
                                break
                        if forced:
                            break
                if not forced:
                    break
            blocks.append(list(kblocks[best_b]))
            chosen_set.add(best_b)
            for i in block_tsubs[best_b]:
                if not covered[i]:
                    w = weights[i]
                    if w != 0.0:
                        covered[i] = True
                        weights[i] = 0.0
                        for b2 in tsub_blocks[i]:
                            block_score[b2] -= w
            if all(covered):
                break
            if decay is not None:
                for i in range(T):
                    if not covered[i]:
                        weights[i] *= decay
                for b2 in range(B):
                    block_score[b2] *= decay
        return blocks

    # Deterministic schedules
    for decay in (None, 0.85, 0.95, 0.7, 0.9, 0.8):
        if time.time() > deadline:
            break
        weights_init = [1.0] * T
        blocks = greedy(weights_init, decay)
        if best_len is None or len(blocks) < best_len:
            best_len = len(blocks)
            best = blocks
        if best_len <= 264:
            break

    # Randomized restarts
    if time.time() < deadline:
        base_seed = 98765
        for it in range(5000):
            if time.time() > deadline:
                break
            random.seed(base_seed + it)
            weights_init = [random.uniform(0.5, 1.5) for _ in range(T)]
            blocks = greedy(weights_init, None)
            if best_len is None or len(blocks) < best_len:
                best_len = len(blocks)
                best = blocks
            if best_len <= 264:
                break

    return best
