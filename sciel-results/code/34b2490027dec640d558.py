"""Combining precomputed-incidence heap greedy + count-based local search + 2-for-1 merge."""
import itertools
import time
import random
import heapq
from collections import defaultdict


def _precompute(v, k, t):
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)

    all_tsubsets = list(itertools.combinations(points, t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    block_tsubs = [None] * n_blocks
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    for bi, block in enumerate(all_blocks):
        tsubs = [tsub_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsubs[bi] = tsubs
        for ts_idx in tsubs:
            tsub_blocks[ts_idx].append(bi)

    # set versions for fast intersection in swap search
    tsub_block_sets = [set(lst) for lst in tsub_blocks]
    return all_blocks, n_blocks, n_tsubsets, block_tsubs, tsub_blocks, tsub_block_sets


def _greedy_cover(seed, deadline, all_blocks, n_blocks, n_tsubsets, block_tsubs, tsub_blocks):
    """Max-coverage greedy via successive gain-augmenting updates (heap priority queue)."""
    rng = random.Random(seed)
    uncovered = [True] * n_tsubsets
    gain = [0] * n_blocks
    for bi in range(n_blocks):
        gain[bi] = len(block_tsubs[bi])

    # Lazy max-heap: entries are (-gain, tiebreak, bi). We verify freshness on pop.
    heap = [(-gain[bi], rng.random(), bi) for bi in range(n_blocks)]
    heapq.heapify(heap)

    selected = []
    while True:
        if time.time() > deadline:
            break
        best_bi = -1
        max_gain = 0
        while heap:
            neg_g, _, bi = heapq.heappop(heap)
            g = -neg_g
            if g == gain[bi] and g > 0:
                best_bi = bi
                max_gain = g
                break
        else:
            break
        if max_gain <= 0:
            break

        selected.append(best_bi)
        # Augment coverage: mark covered t-subsets, decrement gains of blocks covering them.
        for ts_idx in block_tsubs[best_bi]:
            if uncovered[ts_idx]:
                uncovered[ts_idx] = False
                for bj in tsub_blocks[ts_idx]:
                    gain[bj] -= 1
                    heapq.heappush(heap, (-gain[bj], rng.random(), bj))
        gain[best_bi] = -1  # prevent re-selection
        heapq.heappush(heap, (-1, 0, best_bi))

    return selected


def _local_search(blocks, deadline, n_tsubsets, block_tsubs, tsub_blocks, tsub_block_sets, seed):
    """Count-based redundancy removal + 1-for-1 swap + 2-for-1 merge."""
    rng = random.Random(seed)
    block_set = set(blocks)
    cover_count = [0] * n_tsubsets
    for bi in block_set:
        for ts_idx in block_tsubs[bi]:
            cover_count[ts_idx] += 1

    def try_remove(bi):
        for ts_idx in block_tsubs[bi]:
            if cover_count[ts_idx] == 1:
                return False
        block_set.discard(bi)
        if bi in block_set:
            for ts_idx in block_tsubs[bi]:
                cover_count[ts_idx] -= 1
        return True

    improved = True
    while improved and time.time() < deadline:
        improved = False

        # Phase 1: remove all redundant blocks (count-based, cached cover_count)
        to_remove = []
        for bi in list(block_set):
            redundant = True
            for ts_idx in block_tsubs[bi]:
                if cover_count[ts_idx] == 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(bi)
        if to_remove:
            for bi in to_remove:
                block_set.discard(bi)
                for ts_idx in block_tsubs[bi]:
                    cover_count[ts_idx] -= 1
            improved = True
            continue

        # Phase 2: 1-for-1 swap (replace bi with a block covering all its unique t-subsets)
        current_blocks = list(block_set)
        rng.shuffle(current_blocks)
        did_swap = False
        for bi in current_blocks:
            if time.time() > deadline:
                break
            if bi not in block_set:
                continue
            unique_tsubs = [ts_idx for ts_idx in block_tsubs[bi] if cover_count[ts_idx] == 1]
            if not unique_tsubs:
                # redundant, should have been caught
                try_remove(bi)
                improved = True
                did_swap = True
                break
            # candidate blocks covering all unique_tsubs
            min_ts = min(unique_tsubs, key=lambda x: len(tsub_blocks[x]))
            candidates = set(tsub_block_sets[min_ts])
            for ts_idx in unique_tsubs[1:]:
                if not candidates:
                    break
                candidates &= tsub_block_sets[ts_idx]
            candidates.discard(bi)
            if not candidates:
                continue
            cand_list = [c for c in candidates if c not in block_set]
            if not cand_list:
                continue
            rng.shuffle(cand_list)
            for bj in cand_list[:20]:
                # perform swap
                block_set.discard(bi)
                for ts_idx in block_tsubs[bi]:
                    cover_count[ts_idx] -= 1
                block_set.add(bj)
                for ts_idx in block_tsubs[bj]:
                    cover_count[ts_idx] += 1
                improved = True
                did_swap = True
                break
            if did_swap:
                break
        if did_swap:
            continue

        # Phase 3: 2-for-1 merge (replace bi, bj with bk covering all unique t-subsets of both)
        current_blocks = list(block_set)
        rng.shuffle(current_blocks)
        did_merge = False
        for i, bi in enumerate(current_blocks):
            if time.time() > deadline:
                break
            if bi not in block_set:
                continue
            if len(block_set) < 2:
                break
            unique_bi = [ts_idx for ts_idx in block_tsubs[bi] if cover_count[ts_idx] == 1]
            if not unique_bi:
                try_remove(bi)
                improved = True
                did_merge = True
                break
            for bj in current_blocks[i + 1:]:
                if bj not in block_set or bj == bi:
                    continue
                # unique t-subsets of bj given bi still present
                unique_bj = [ts_idx for ts_idx in block_tsubs[bj] if cover_count[ts_idx] == 1]
                if not unique_bj:
                    # bj redundant
                    try_remove(bj)
                    improved = True
                    did_merge = True
                    break
                combined = set(unique_bi) | set(unique_bj)
                # candidate blocks covering all combined t-subsets
                min_ts = min(combined, key=lambda x: len(tsub_blocks[x]))
                candidates = set(tsub_block_sets[min_ts])
                for ts_idx in combined:
                    if not candidates:
                        break
                    candidates &= tsub_block_sets[ts_idx]
                candidates.discard(bi)
                candidates.discard(bj)
                if not candidates:
                    continue
                cand_list = [c for c in candidates if c not in block_set]
                if not cand_list:
                    continue
                rng.shuffle(cand_list)
                for bk in cand_list[:15]:
                    block_set.discard(bi)
                    block_set.discard(bj)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                    for ts_idx in block_tsubs[bj]:
                        cover_count[ts_idx] -= 1
                    block_set.add(bk)
                    for ts_idx in block_tsubs[bk]:
                        cover_count[ts_idx] += 1
                    improved = True
                    did_merge = True
                    break
                if did_merge:
                    break
            if did_merge:
                break
        if did_merge:
            continue

    return list(block_set)


def construct(v, k, t):
    start_time = time.time()
    hard_deadline = start_time + 115.0  # safety margin before 150s kill

    all_blocks, n_blocks, n_tsubsets, block_tsubs, tsub_blocks, tsub_block_sets = _precompute(v, k, t)

    best_blocks = None
    best_size = float('inf')

    # Multi-start greedy, reserving time for local search.
    # Budget: spend most time on greedy restarts, then local search on the best.
    n_starts = 12
    greedy_deadline = hard_deadline - 40.0  # reserve ~40s for local search

    for s in range(n_starts):
        if time.time() > greedy_deadline:
            break
        sel = _greedy_cover(s, greedy_deadline, all_blocks, n_blocks, n_tsubsets, block_tsubs, tsub_blocks)
        # Convert to distinct blocks (greedy never re-selects, but guard)
        seen = set()
        distinct = []
        for bi in sel:
            if bi not in seen:
                seen.add(bi)
                distinct.append(bi)
        if len(distinct) < best_size:
            best_size = len(distinct)
            best_blocks = distinct

    if best_blocks is None:
        # Fallback: simple greedy with no time pressure
        sel = _greedy_cover(0, hard_deadline - 5, all_blocks, n_blocks, n_tsubsets, block_tsubs, tsub_blocks)
        best_blocks = list(dict.fromkeys(sel))

    # Local search on the best solution, with a couple of restarts if time permits.
    if time.time() < hard_deadline - 5:
        best_blocks = _local_search(best_blocks, hard_deadline - 2, n_tsubsets, block_tsubs, tsub_blocks, tsub_block_sets, seed=0)

    # Convert block indices back to point tuples
    result = [list(all_blocks[bi]) for bi in best_blocks]
    return result
