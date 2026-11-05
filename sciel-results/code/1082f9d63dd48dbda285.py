"""Covering design C(24,7,3): minimize number of 7-subsets of 24 points
so that every 3-subset is contained in some block.
Entry point: construct(v, k, t) -> list of blocks (sorted k-subsets)."""

import itertools
import time
import random
import numpy as np

# Precompute all 3-subsets of range(v) as integer-encoded triples for speed.
# For a given v, t we encode each t-subset as a sorted tuple of ints.

def _all_tsubsets(v, t):
    return [tuple(c) for c in itertools.combinations(range(v), t)]


def _block_to_triples(block, t):
    return list(itertools.combinations(block, t))


def _greedy_cover(v, k, t, seed=None, ratio_cost=True):
    """Greedy set-cover: repeatedly open a block on the first uncovered t-subset,
    then extend by choosing the point that maximizes new t-subsets covered per cost.
    Returns list of blocks."""
    rng = random.Random(seed)
    all_triples = _all_tsubsets(v, t)
    uncovered = set(all_triples)
    blocks = []
    points = list(range(v))

    # Precompute, for each point, which triples contain it, to speed gain.
    # Actually gain depends on current block; we compute on the fly but vectorized-ish.
    while uncovered:
        # start block with the first (lexicographically) uncovered triple's points
        start = min(uncovered)
        block = list(start)
        in_block = set(block)
        # Precompute the (t-1)-subsets of the current block that we extend
        while len(block) < k:
            # For each candidate point not in block, count how many uncovered triples
            # would be completed: triples = (t-1)-subset of block + x that are uncovered.
            best_x = None
            best_score = -1
            best_count = -1
            # candidates
            for x in points:
                if x in in_block:
                    continue
                # count uncovered triples of form (comb of t-1 from block) + x
                cnt = 0
                for rest in itertools.combinations(block, t - 1):
                    tr = tuple(sorted(rest + (x,)))
                    if tr in uncovered:
                        cnt += 1
                if cnt == 0:
                    continue
                if ratio_cost:
                    # cost: 1 + fraction of block points that already co-occur heavily
                    # simple penalty: prefer points that create more *new* coverage per point
                    score = cnt / (1.0 + 0.0)  # keep ratio form; cost=1 baseline
                    # stronger ratio: divide by (number of existing points sharing a pair)
                    # approximate redundancy: how many existing block points are 'close'
                    # use a mild penalty based on block size growth
                    score = cnt / (1.0 + 0.15 * (len(block)))
                else:
                    score = cnt
                if score > best_score or (score == best_score and cnt > best_count):
                    best_score = score
                    best_count = cnt
                    best_x = x
            if best_x is None:
                # no candidate adds coverage; pick an arbitrary point to fill
                best_x = next(p for p in points if p not in in_block)
                best_count = 0
            block.append(best_x)
            in_block.add(best_x)
        block.sort()
        blocks.append(block)
        # remove all triples contained in this block
        for tr in itertools.combinations(block, t):
            uncovered.discard(tr)
    return blocks


def _is_valid(blocks, v, t):
    """Check every t-subset is covered."""
    covered = set()
    for b in blocks:
        for tr in itertools.combinations(b, t):
            covered.add(tr)
    return len(covered) == len(list(itertools.combinations(range(v), t)))


def _local_improve(blocks, v, k, t, max_rounds=40, deadline=None):
    """Try to reduce block count by dropping a block if redundant, or by
    replacing a block with one that covers more uncovered triples.
    Only accept changes that keep validity and do not increase block count."""
    all_triples = _all_tsubsets(v, t)
    # coverage count per triple
    cov = {tr: 0 for tr in all_triples}
    block_triples = []
    for b in blocks:
        trs = list(itertools.combinations(b, t))
        block_triples.append(set(trs))
        for tr in trs:
            cov[tr] += 1

    improved = True
    rounds = 0
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1
        if deadline is not None and time.time() > deadline:
            break
        # 1) drop a redundant block (every triple it covers is covered by >=2 blocks)
        for i, bset in enumerate(block_triples):
            redundant = True
            for tr in bset:
                if cov[tr] <= 1:
                    redundant = False
                    break
            if redundant:
                for tr in bset:
                    cov[tr] -= 1
                block_triples.pop(i)
                blocks.pop(i)
                improved = True
                break
        # 2) try to merge: not straightforward for fixed k; skip.
        # 3) try replacing a block with a better one: pick the block whose removal
        #    leaves the most uncovered triples, and rebuild it greedily.
        if not improved:
            # find block whose removal maximizes newly-uncovered triples
            best_i = -1
            best_uncovered = 0
            for i, bset in enumerate(block_triples):
                unc = sum(1 for tr in bset if cov[tr] == 1)
                if unc > best_uncovered:
                    best_uncovered = unc
                    best_i = i
            if best_i >= 0 and best_uncovered > 0:
                # remove it
                for tr in block_triples[best_i]:
                    cov[tr] -= 1
                removed_block = blocks.pop(best_i)
                block_triples.pop(best_i)
                # rebuild greedily starting from most uncovered triple
                uncovered = {tr for tr in all_triples if cov[tr] == 0}
                if uncovered:
                    start = min(uncovered)
                    newblock = list(start)
                    inb = set(newblock)
                    while len(newblock) < k:
                        best_x, best_cnt = None, 0
                        for x in range(v):
                            if x in inb:
                                continue
                            cnt = 0
                            for rest in itertools.combinations(newblock, t - 1):
                                tr = tuple(sorted(rest + (x,)))
                                if tr in uncovered:
                                    cnt += 1
                            if cnt > best_cnt:
                                best_cnt = cnt
                                best_x = x
                        if best_x is None:
                            best_x = next(p for p in range(v) if p not in inb)
                        newblock.append(best_x)
                        inb.add(best_x)
                    newblock.sort()
                    newset = set(itertools.combinations(newblock, t))
                    # only accept if it covers at least as many previously-uncovered
                    covered_new = 0
                    for tr in newset:
                        if cov[tr] == 0:
                            covered_new += 1
                    if covered_new >= best_uncovered:
                        blocks.append(newblock)
                        block_triples.append(newset)
                        for tr in newset:
                            cov[tr] += 1
                        improved = True
                    else:
                        # revert: put old block back
                        blocks.append(removed_block)
                        block_triples.append(block_triples and set(itertools.combinations(removed_block, t)) or set(itertools.combinations(removed_block, t)))
                        for tr in block_triples[-1]:
                            cov[tr] += 1
    return blocks


def construct(v, k, t):
    start = time.time()
    deadline = start + 120.0  # safety margin before 150s kill
    # Fast path: if small, just do one greedy.
    best = None
    best_len = None

    # Deterministic first attempt
    cand = _greedy_cover(v, k, t, seed=0, ratio_cost=True)
    if best_len is None or len(cand) < best_len:
        best = cand
        best_len = len(cand)

    # Local improvement on the best so far
    improved = _local_improve(best, v, k, t, deadline=deadline)
    if len(improved) < best_len:
        best = improved
        best_len = len(best)

    # Multiple restarts with different seeds while time allows
    attempt = 1
    while time.time() < deadline - 5.0:
        cand = _greedy_cover(v, k, t, seed=attempt, ratio_cost=True)
        if len(cand) < best_len:
            cand = _local_improve(cand, v, k, t, deadline=deadline)
            if len(cand) < best_len:
                best = cand
                best_len = len(cand)
        attempt += 1
        if attempt > 60:  # cap restarts to avoid pathological loops
            break

    # Final validity safety check; if invalid, fall back to exhaustive-ish greedy
    if best is None or not _is_valid(best, v, t):
        best = _greedy_cover(v, k, t, seed=12345, ratio_cost=False)

    # Ensure blocks are distinct and sorted
    seen = set()
    out = []
    for b in best:
        key = tuple(sorted(b))
        if key not in seen:
            seen.add(key)
            out.append(list(key))
    return out
