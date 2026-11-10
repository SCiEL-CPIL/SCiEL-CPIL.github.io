"""Bitmask-accelerated randomized greedy covering with pruning and deadline guard."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
from collections import defaultdict


def construct(v, k, t):
    deadline = time.time() + 118.0
    rng = random.Random(12345)
    points = list(range(v))

    # --- precompute t-subsets and their bitmask signatures -------------------
    tsubs = list(itertools.combinations(range(v), t))
    tsub_masks = []
    for c in tsubs:
        m = 0
        for x in c:
            m |= (1 << x)
        tsub_masks.append(m)
    n_t = len(tsubs)

    # --- greedy build of one block -------------------------------------------
    def greedy_block(block, uncovered):
        cur = list(block)
        while len(cur) < k:
            best = -1
            bestx = None
            for x in points:
                if x in cur:
                    continue
                # count uncovered t-subsets covered by adding x
                g = 0
                for comb in itertools.combinations(cur, t - 1):
                    m = 0
                    for y in comb:
                        m |= (1 << y)
                    m |= (1 << x)
                    if m in uncovered:
                        g += 1
                if g > best:
                    best = g
                    bestx = x
            if bestx is None:
                break
            cur.append(bestx)
        cur.sort()
        return cur

    def build_one():
        uncovered = set(tsub_masks)
        blocks = []
        while uncovered and time.time() < deadline:
            # pick an uncovered t-subset (random among a small sample for variety)
            sample = rng.sample(uncovered, min(32, len(uncovered)))
            seed_mask = max(sample)
            # find one t-subset with that mask to seed the block
            seed = None
            for i, m in enumerate(tsub_masks):
                if m == seed_mask:
                    seed = tsubs[i]
                    break
            block = greedy_block(list(seed), uncovered)
            block_set = set(block)
            blocks.append(block)
            # remove all t-subsets contained in this block
            for comb in itertools.combinations(block, t):
                m = 0
                for y in comb:
                    m |= (1 << y)
                uncovered.discard(m)
        return blocks, uncovered

    # --- pruning: remove redundant blocks ------------------------------------
    def prune(blocks):
        # for each block, the set of t-subsets it covers
        block_covers = []
        for b in blocks:
            s = set()
            for comb in itertools.combinations(b, t):
                m = 0
                for y in comb:
                    m |= (1 << y)
                s.add(m)
            block_covers.append(s)
        # union of all
        union = set()
        for s in block_covers:
            union |= s
        # a block is redundant if its covered set is a subset of the union of others
        kept = []
        kept_index = []
        for i, s in enumerate(block_covers):
            if time.time() >= deadline:
                break
            others = set()
            for j, s2 in enumerate(block_covers):
                if j == i:
                    continue
                others |= s2
            if s.issubset(others):
                continue  # redundant
            kept.append(blocks[i])
            kept_index.append(i)
        # rebuild covers for kept only and do a second pass
        if len(kept) < len(blocks):
            covers2 = []
            for b in kept:
                s = set()
                for comb in itertools.combinations(b, t):
                    m = 0
                    for y in comb:
                        m |= (1 << y)
                    s.add(m)
                covers2.append(s)
            final = []
            for i, s in enumerate(covers2):
                if time.time() >= deadline:
                    break
                others = set()
                for j, s2 in enumerate(covers2):
                    if j == i:
                        continue
                    others |= s2
                if not s.issubset(others):
                    final.append(kept[i])
            return final
        return kept

    best = None
    # multi-restart greedy; keep the smallest valid (or smallest) block list
    restart = 0
    while time.time() < deadline - 25.0:
        blocks, uncovered = build_one()
        if not blocks:
            break
        if not uncovered:
            # valid covering; try to prune to reduce size
            pruned = prune(blocks)
            if best is None or len(pruned) < len(best):
                best = pruned
            # if we have a valid small one, we can still try more restarts
        else:
            # not fully covered; still keep as a candidate only if smaller so far
            if best is None or len(blocks) < len(best):
                best = blocks
        restart += 1
        if time.time() >= deadline - 25.0:
            break

    if best is None:
        # fallback: run one more full greedy to guarantee something
        blocks, _ = build_one()
        best = blocks

    return best
# EVOLVE-BLOCK-END
