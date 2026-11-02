import itertools
import time
import random
import numpy as np
from collections import defaultdict

def _precompute(v, k, t):
    tsubs = list(itertools.combinations(range(v), t))
    tsub_id = {ts: i for i, ts in enumerate(tsubs)}
    block_tsubs = {}
    for block in itertools.combinations(range(v), k):
        covered = np.array([tsub_id[ts] for ts in itertools.combinations(block, t)], dtype=np.int32)
        block_tsubs[block] = covered
    return tsubs, tsub_id, block_tsubs

def _greedy_cover(v, k, t, rng, tsubs, tsub_id, block_tsubs, candidates):
    n_tsubs = len(tsubs)
    uncovered = np.ones(n_tsubs, dtype=np.int32)
    blocks = []
    for block in candidates:
        covered = block_tsubs[block]
        gain = int(np.sum(uncovered[covered]))
        if gain <= 0:
            continue
        blocks.append(block)
        uncovered[covered] = 0
        if not uncovered.any():
            break
    return blocks

def _dual_greedy_cover(v, k, t, rng, tsubs, tsub_id, block_tsubs, candidates, deadline):
    n_tsubs = len(tsubs)
    cov_count = np.zeros(n_tsubs, dtype=np.int32)
    blocks = []
    for block in candidates:
        covered = block_tsubs[block]
        gain = int(np.sum(cov_count[covered] == 0))
        if gain <= 0:
            continue
        blocks.append(block)
        cov_count[covered] += 1
        if not (cov_count > 0).all():
            continue
        if time.time() >= deadline:
            break
    return blocks

def _local_search(blocks, v, k, t, deadline, tsubs, tsub_id, block_tsubs, candidates):
    blocks = [list(b) for b in blocks]
    n_tsubs = len(tsubs)

    def build_cov():
        cov = np.zeros((n_tsubs, len(blocks)), dtype=bool)
        for bi, block in enumerate(blocks):
            cov[block_tsubs[tuple(block)][:, None], bi] = True
        return cov

    def unique_sets(bi, cov):
        col = cov[:, bi]
        cnt = np.sum(cov, axis=1)
        return np.where(col & (cnt == 1))[0]

    def try_merge(bi, bj, cov):
        u = np.where((cov[:, bi] | cov[:, bj]) & (np.sum(cov, axis=1) == 1))[0]
        elems = set()
        for ts_id in u:
            elems.update(tsubs[int(ts_id)])
        if len(elems) > k:
            return None
        remaining = [x for x in range(v) if x not in elems]
        needed = k - len(elems)
        if needed == 0:
            cand = tuple(sorted(elems))
            if cand in block_tsubs:
                return [cand]
        elif needed <= 2:
            for ext in itertools.combinations(remaining, needed):
                cand = tuple(sorted(elems | set(ext)))
                if cand in block_tsubs:
                    return [cand]
        return None

    def try_swap(bi, cand, cov):
        u = unique_sets(bi, cov)
        if len(u) == 0:
            return True
        elems = set()
        for ts_id in u:
            elems.update(tsubs[int(ts_id)])
        if len(elems) > k:
            return False
        cand_set = set(cand)
        if not elems.issubset(cand_set):
            return False
        return True

    cov = build_cov()
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(blocks)

        # Removal pass
        for bi in range(n):
            if time.time() >= deadline:
                return blocks
            if len(unique_sets(bi, cov)) == 0:
                blocks.pop(bi)
                cov = build_cov()
                improved = True
                break
        if improved:
            continue

        # Merge pass
        for bi in range(len(blocks)):
            if time.time() >= deadline:
                return blocks
            for bj in range(bi + 1, len(blocks)):
                if time.time() >= deadline:
                    return blocks
                merged = try_merge(bi, bj, cov)
                if merged is not None:
                    blocks.pop(bj)
                    blocks.pop(bi)
                    blocks.insert(bi, list(merged[0]))
                    cov = build_cov()
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue

        # Swap pass: replace one block with a candidate that covers all its unique t-subsets
        for bi in range(len(blocks)):
            if time.time() >= deadline:
                return blocks
            u = unique_sets(bi, cov)
            if len(u) == 0:
                continue
            elems = set()
            for ts_id in u:
                elems.update(tsubs[int(ts_id)])
            if len(elems) > k:
                continue
            # candidate must be a k-block containing elems
            remaining = [x for x in range(v) if x not in elems]
            needed = k - len(elems)
            found = False
            for ext in itertools.combinations(remaining, needed):
                cand = tuple(sorted(elems | set(ext)))
                if cand in block_tsubs and cand != tuple(blocks[bi]):
                    if try_swap(bi, cand, cov):
                        blocks[bi] = list(cand)
                        cov = build_cov()
                        improved = True
                        found = True
                        break
            if found:
                break
    return blocks

def construct(v, k, t):
    start = time.time()
    deadline = start + 145
    rng = random.Random(42)

    tsubs, tsub_id, block_tsubs = _precompute(v, k, t)
    candidates = list(block_tsubs.keys())

    best_blocks = None
    best_count = float('inf')

    # Phase 1: diverse randomized greedy restarts
    greedy_deadline = deadline - 90
    n_cand = len(candidates)
    while time.time() < greedy_deadline:
        order = list(candidates)
        rng.shuffle(order)
        blocks = _greedy_cover(v, k, t, rng, tsubs, tsub_id, block_tsubs, order)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
        # also try dual-greedy on a few orders
        order2 = list(candidates)
        rng.shuffle(order2)
        blocks2 = _dual_greedy_cover(v, k, t, rng, tsubs, tsub_id, block_tsubs, order2, greedy_deadline)
        if len(blocks2) < best_count:
            best_count = len(blocks2)
            best_blocks = blocks2

    # Phase 2: local search on the best
    if best_blocks is not None and time.time() < deadline - 3:
        best_blocks = _local_search(best_blocks, v, k, t, deadline - 2, tsubs, tsub_id, block_tsubs, candidates)

    return best_blocks
