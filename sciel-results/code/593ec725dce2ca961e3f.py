"""Covering design C(19,7,4): minimize number of 7-subsets covering all 4-subsets.
Strategy: fast bitset greedy construction + aggressive local search (removal/swap/add).
"""
import time
import itertools
import random

_DEADLINE = 118.0  # seconds, safety margin under 150s
_START = None


def _elapsed():
    return time.time() - _START


def _precompute(v, k, t):
    """Precompute all t-subsets and for every k-subset its coverage as an int bitmask."""
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_index = {c: i for i, c in enumerate(all_t)}
    k_subsets = list(itertools.combinations(range(v), k))
    # coverage: list of int bitmasks for each k-subset
    coverage = []
    for ks in k_subsets:
        mask = 0
        for ts in itertools.combinations(ks, t):
            mask |= (1 << t_index[ts])
        coverage.append(mask)
    full_mask = (1 << n_t) - 1
    return all_t, t_index, k_subsets, coverage, full_mask


def _popcount(x):
    return bin(x).count('1')


def _greedy_construct(v, k, t, rng):
    """Greedy set cover using bitmask operations."""
    all_t, t_index, k_subsets, coverage, full_mask = _precompute(v, k, t)
    n_t = len(all_t)
    uncovered = full_mask
    blocks = []
    n_ks = len(k_subsets)
    while uncovered and _elapsed() < _DEADLINE - 25:
        best_i = -1
        best_gain = 0
        for i in range(n_ks):
            g = _popcount(coverage[i] & uncovered)
            if g > best_gain:
                best_gain = g
                best_i = i
        if best_i < 0 or best_gain == 0:
            break
        blocks.append(best_i)
        uncovered &= ~coverage[best_i]
    return blocks, all_t, t_index, k_subsets, coverage, full_mask


def _local_search(blocks, coverage, n_t, k_subsets, rng, full_mask, time_budget):
    """Aggressive local search: repeated removal + swap passes with bitmask speed."""
    blocks = list(blocks)
    n_ks = len(k_subsets)

    def union_mask(exclude_idx=None):
        m = 0
        for idx, b in enumerate(blocks):
            if exclude_idx is not None and idx == exclude_idx:
                continue
            m |= coverage[b]
        return m

    improved = True
    while improved and _elapsed() < time_budget:
        improved = False
        # --- Removal pass (multiple iterations until no more removals) ---
        while _elapsed() < time_budget:
            removed_any = False
            i = 0
            while i < len(blocks) and _elapsed() < time_budget:
                m = union_mask(exclude_idx=i)
                if m == full_mask:
                    blocks.pop(i)
                    removed_any = True
                    improved = True
                    continue
                i += 1
            if not removed_any:
                break

        # --- Swap pass: try random swaps that enable removals ---
        n_swaps = 60
        for _ in range(n_swaps):
            if _elapsed() >= time_budget:
                break
            i = rng.randrange(len(blocks))
            j = rng.randrange(n_ks)
            if j in blocks:
                continue
            old = blocks[i]
            blocks[i] = j
            # Try to find a removable block after swap
            removed = False
            for idx in range(len(blocks)):
                m = union_mask(exclude_idx=idx)
                if m == full_mask:
                    blocks.pop(idx)
                    removed = True
                    improved = True
                    break
            if not removed:
                blocks[i] = old
            # After a successful swap+remove, do another removal pass
            if removed:
                while _elapsed() < time_budget:
                    removed_any = False
                    ii = 0
                    while ii < len(blocks) and _elapsed() < time_budget:
                        m = union_mask(exclude_idx=ii)
                        if m == full_mask:
                            blocks.pop(ii)
                            removed_any = True
                            improved = True
                            continue
                        ii += 1
                    if not removed_any:
                        break

        # --- Repair if somehow not covered ---
        m = 0
        for b in blocks:
            m |= coverage[b]
        if m != full_mask:
            missing = full_mask & ~m
            best_i = -1
            best_g = 0
            for idx in range(n_ks):
                g = _popcount(coverage[idx] & missing)
                if g > best_g:
                    best_g = g
                    best_i = idx
            if best_i >= 0:
                blocks.append(best_i)

    return blocks


def construct(v, k, t):
    global _START
    _START = time.time()
    rng = random.Random(12345)

    all_t, t_index, k_subsets, coverage, full_mask = _precompute(v, k, t)
    n_t = len(all_t)

    # Greedy construction
    blocks, _, _, _, _, _ = _greedy_construct(v, k, t, rng)

    # Local search with most of the time budget
    time_budget = _DEADLINE - 3
    blocks = _local_search(blocks, coverage, n_t, k_subsets, rng, full_mask, time_budget)

    # Safety: ensure valid
    m = 0
    for b in blocks:
        m |= coverage[b]
    if m != full_mask:
        while m != full_mask and _elapsed() < _DEADLINE - 1:
            missing = full_mask & ~m
            best_i = -1
            best_g = 0
            for idx in range(len(k_subsets)):
                g = _popcount(coverage[idx] & missing)
                if g > best_g:
                    best_g = g
                    best_i = idx
            if best_i < 0:
                break
            blocks.append(best_i)
            m |= coverage[best_i]

    # Return as list of k-subsets (lists of ints)
    result = [list(k_subsets[b]) for b in blocks]
    # Deduplicate
    seen = set()
    dedup = []
    for blk in result:
        key = tuple(sorted(blk))
        if key not in seen:
            seen.add(key)
            dedup.append(list(key))
    return dedup
