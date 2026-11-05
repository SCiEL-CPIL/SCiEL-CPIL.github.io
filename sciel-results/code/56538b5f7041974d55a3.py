"""Covering design C(v,k,t) solver: fast bitmask greedy + randomized local search."""
import time
import random
import itertools

import numpy as np


def construct(v, k, t):
    start = time.time()
    deadline = start + 115.0          # safety margin under 120s limit
    rng = random.Random(20240517)

    # Precompute all t-subsets of the v points, indexed 0..N-1.
    tris = list(itertools.combinations(range(v), t))
    N = len(tris)
    tri_index = {tr: i for i, tr in enumerate(tris)}
    full = (1 << N) - 1

    # Block coverage bitmask for every k-subset.
    block_tris = {}
    for blk in itertools.combinations(range(v), k):
        mask = 0
        for tr in itertools.combinations(blk, t):
            mask |= 1 << tri_index[tr]
        block_tris[blk] = mask
    all_keys = list(block_tris.keys())
    all_masks = [block_tris[bk] for bk in all_keys]
    Nblocks = len(all_keys)

    # Greedy seed (fast bitmask version).
    uncovered = full
    blocks = []
    while uncovered:
        best_i = -1
        best_c = -1
        for i, m in enumerate(all_masks):
            c = (m & uncovered).bit_count()
            if c > best_c:
                best_c = c
                best_i = i
        blocks.append(all_keys[best_i])
        uncovered &= ~all_masks[best_i]
        if time.time() > deadline - 15.0:
            break

    def make_block_list(blk_list):
        return [sorted(b) for b in blk_list]

    best = make_block_list(blocks)

    if uncovered == 0 and Nblocks > 1:
        # Randomized local search to shrink the block count.
        cur = [list(b) for b in best]
        cur_masks = [block_tris[tuple(b)] for b in cur]
        cur_union = 0
        for m in cur_masks:
            cur_union |= m
        cur_covers_all = (cur_union == full)
        best_count = len(cur)

        def union_of(lst):
            u = 0
            for m in lst:
                u |= m
            return u

        while time.time() < deadline - 1.0:
            # Attempt removal of a redundant block.
            if cur_covers_all and len(cur) >= 1:
                for i in range(len(cur)):
                    if union_of(cur_masks[:i] + cur_masks[i+1:]) == full:
                        cur.pop(i)
                        cur_masks.pop(i)
                        cur_union = union_of(cur_masks)
                        cur_covers_all = True
                        if len(cur) < best_count:
                            best_count = len(cur)
                            best = make_block_list(cur)
                        break
            # Attempt a swap: replace one block to allow a later removal / net gain.
            if len(cur) >= 1 and time.time() < deadline - 3.0:
                i = rng.randrange(len(cur))
                rmask = cur_masks[i]
                u = union_of(cur_masks[:i] + cur_masks[i+1:])
                deficit = rmask & ~u
                if deficit == 0:
                    cur.pop(i)
                    cur_masks.pop(i)
                    cur_union = u
                    cur_covers_all = True
                    if len(cur) < best_count:
                        best_count = len(cur)
                        best = make_block_list(cur)
                    continue
                # find a replacement covering the deficit
                cand = [j for j, m in enumerate(all_masks)
                        if m & deficit == deficit and m != rmask]
                if cand:
                    j = cand[rng.randrange(len(cand))]
                    cur[i] = list(all_keys[j])
                    cur_masks[i] = all_masks[j]
                    cur_union = u | all_masks[j]
                    cur_covers_all = (cur_union == full)
                    if len(cur) < best_count and cur_covers_all:
                        best_count = len(cur)
                        best = make_block_list(cur)

    # Final validity check; if invalid, rebuild greedily to guarantee coverage.
    if best:
        cov = 0
        for b in best:
            cov |= block_tris[tuple(b)]
        if cov != full:
            uncovered = full
            blocks = []
            while uncovered:
                best_i = -1
                best_c = -1
                for i, m in enumerate(all_masks):
                    c = (m & uncovered).bit_count()
                    if c > best_c:
                        best_c = c
                        best_i = i
                blocks.append(all_keys[best_i])
                uncovered &= ~all_masks[best_i]
            best = make_block_list(blocks)

    return best
