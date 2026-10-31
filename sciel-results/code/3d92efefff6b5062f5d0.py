"""Covering design C(19,7,4): minimize number of 7-subsets covering all 4-subsets.
Strategy: fast bitset greedy construction + coverage-count local search with removal,
swap, and 2-swap moves.
"""
import time
import itertools
import random

_DEADLINE = 115.0  # seconds, safety margin under 120s
_START = None


def _elapsed():
    return time.time() - _START


def _precompute(v, k, t):
    """Precompute all t-subsets as ints, and for every k-subset its coverage bitmap."""
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_index = {c: i for i, c in enumerate(all_t)}
    k_subsets = list(itertools.combinations(range(v), k))
    coverage_bits = []
    for ks in k_subsets:
        mask = 0
        for ts in itertools.combinations(ks, t):
            idx = t_index[ts]
            mask |= (1 << idx)
        coverage_bits.append(mask)
    return all_t, t_index, k_subsets, coverage_bits


def _greedy_construct(v, k, t, rng):
    """Greedy set cover: repeatedly add the k-subset covering the most uncovered t-subsets."""
    all_t, t_index, k_subsets, coverage_bits = _precompute(v, k, t)
    n_t = len(all_t)
    uncovered = (1 << n_t) - 1
    blocks = []
    n_k = len(k_subsets)

    while uncovered and _elapsed() < _DEADLINE - 25:
        best_i = -1
        best_gain = 0
        for i in range(n_k):
            g = (coverage_bits[i] & uncovered).bit_count()
            if g > best_gain:
                best_gain = g
                best_i = i
        if best_i < 0:
            break
        blocks.append(best_i)
        uncovered &= ~coverage_bits[best_i]

    return blocks, all_t, t_index, k_subsets, coverage_bits


def _local_search(blocks, coverage_bits, n_t, k_subsets, rng, time_budget):
    """Local search using coverage counts for O(1) removal checks.
    Moves: removal, single swap (replace one block, try removal), double swap
    (replace two blocks, try removal)."""
    blocks = list(blocks)
    full_mask = (1 << n_t) - 1
    n_k = len(k_subsets)

    def compute_counts(blk_list):
        cnt = [0] * n_t
        for b in blk_list:
            mask = coverage_bits[b]
            i = 0
            while mask:
                low = mask & (-mask)
                i = low.bit_length() - 1
                cnt[i] += 1
                mask ^= low
        return cnt

    def try_remove(blk_list, counts, full_mask):
        """Try to remove any redundant block. Returns (new_list, new_counts, removed)."""
        n = len(blk_list)
        for i in range(n):
            mask = coverage_bits[blk_list[i]]
            m = mask
            removable = True
            while m:
                low = m & (-m)
                idx = low.bit_length() - 1
                if counts[idx] <= 1:
                    removable = False
                    break
                m ^= low
            if removable:
                new_list = blk_list[:i] + blk_list[i+1:]
                new_counts = counts[:]
                m = mask
                while m:
                    low = m & (-m)
                    idx = low.bit_length() - 1
                    new_counts[idx] -= 1
                    m ^= low
                return new_list, new_counts, True
        return blk_list, counts, False

    # Initialize coverage counts
    counts = compute_counts(blocks)

    # Verify initial coverage
    cov = 0
    for b in blocks:
        cov |= coverage_bits[b]
    if cov != full_mask:
        # Repair first
        while cov != full_mask and _elapsed() < time_budget:
            missing = full_mask ^ cov
            best_i = -1
            best_g = 0
            for i in range(n_k):
                g = (coverage_bits[i] & missing).bit_count()
                if g > best_g:
                    best_g = g
                    best_i = i
            if best_i < 0:
                break
            blocks.append(best_i)
            cov |= coverage_bits[best_i]
        counts = compute_counts(blocks)

    improved = True
    stall_count = 0
    while improved and _elapsed() < time_budget:
        improved = False
        stall_count += 1

        # --- Removal pass ---
        new_blocks, new_counts, removed = try_remove(blocks, counts, full_mask)
        if removed:
            blocks = new_blocks
            counts = new_counts
            improved = True
            stall_count = 0

        # --- Single swap pass: replace block i with j, then try removal ---
        n_blocks = len(blocks)
        for _ in range(80):
            if _elapsed() >= time_budget:
                break
            if n_blocks == 0:
                break
            i = rng.randrange(n_blocks)
            j = rng.randrange(n_k)
            if j in blocks:
                continue

            old = blocks[i]
            blocks[i] = j

            # Update counts for the swap
            new_counts = counts[:]
            old_mask = coverage_bits[old]
            m = old_mask
            while m:
                low = m & (-m)
                idx = low.bit_length() - 1
                new_counts[idx] -= 1
                m ^= low
            new_mask = coverage_bits[j]
            m = new_mask
            while m:
                low = m & (-m)
                idx = low.bit_length() - 1
                new_counts[idx] += 1
                m ^= low

            # Check coverage is still valid
            valid = True
            m = new_mask
            while m:
                low = m & (-m)
                idx = low.bit_length() - 1
                if new_counts[idx] < 0:
                    valid = False
                    break
                m ^= low

            if valid:
                # Try removal
                res_list, res_counts, res_removed = try_remove(blocks, new_counts, full_mask)
                if res_removed:
                    blocks = res_list
                    counts = res_counts
                    improved = True
                    stall_count = 0
                    n_blocks = len(blocks)
                    continue
            # Revert
            blocks[i] = old

        # --- Double swap pass: replace blocks i and j with a and b, then try removal ---
        if n_blocks >= 2:
            for _ in range(40):
                if _elapsed() >= time_budget:
                    break
                i = rng.randrange(n_blocks)
                j = rng.randrange(n_blocks)
                if i == j:
                    continue
                a = rng.randrange(n_k)
                b = rng.randrange(n_k)
                if a in blocks or b in blocks:
                    continue
                if a == b:
                    continue

                old_i = blocks[i]
                old_j = blocks[j]
                blocks[i] = a
                blocks[j] = b

                new_counts = counts[:]
                for old_b in (old_i, old_j):
                    m = coverage_bits[old_b]
                    while m:
                        low = m & (-m)
                        idx = low.bit_length() - 1
                        new_counts[idx] -= 1
                        m ^= low
                for new_b in (a, b):
                    m = coverage_bits[new_b]
                    while m:
                        low = m & (-m)
                        idx = low.bit_length() - 1
                        new_counts[idx] += 1
                        m ^= low

                valid = True
                for idx in range(n_t):
                    if new_counts[idx] < 0:
                        valid = False
                        break

                if valid:
                    res_list, res_counts, res_removed = try_remove(blocks, new_counts, full_mask)
                    if res_removed:
                        blocks = res_list
                        counts = res_counts
                        improved = True
                        stall_count = 0
                        n_blocks = len(blocks)
                        continue
                # Revert
                blocks[i] = old_i
                blocks[j] = old_j

        # --- If stalled, do a random perturbation: remove a random block, repair ---
        if not improved and stall_count > 3 and n_blocks > 5:
            if _elapsed() < time_budget - 5:
                i = rng.randrange(n_blocks)
                removed_block = blocks.pop(i)
                # Repair
                cov = 0
                for b in blocks:
                    cov |= coverage_bits[b]
                while cov != full_mask and _elapsed() < time_budget - 2:
                    missing = full_mask ^ cov
                    best_i = -1
                    best_g = 0
                    for idx in range(n_k):
                        g = (coverage_bits[idx] & missing).bit_count()
                        if g > best_g:
                            best_g = g
                            best_i = idx
                    if best_i < 0:
                        break
                    blocks.append(best_i)
                    cov |= coverage_bits[best_i]
                counts = compute_counts(blocks)
                n_blocks = len(blocks)
                stall_count = 0

    return blocks


def construct(v, k, t):
    global _START
    _START = time.time()
    rng = random.Random(12345)

    all_t, t_index, k_subsets, coverage_bits = _precompute(v, k, t)
    n_t = len(all_t)

    # Greedy construction
    blocks, _, _, _, _ = _greedy_construct(v, k, t, rng)

    # Local search
    time_budget = _DEADLINE - 5
    blocks = _local_search(blocks, coverage_bits, n_t, k_subsets, rng, time_budget)

    # Safety: ensure valid; if not, do a quick repair
    cov = 0
    for b in blocks:
        cov |= coverage_bits[b]

    full_mask = (1 << n_t) - 1
    if cov != full_mask:
        while cov != full_mask and _elapsed() < _DEADLINE - 2:
            missing = full_mask ^ cov
            best_i = -1
            best_g = 0
            for idx, mask in enumerate(coverage_bits):
                g = (mask & missing).bit_count()
                if g > best_g:
                    best_g = g
                    best_i = idx
            if best_i < 0:
                break
            blocks.append(best_i)
            cov |= coverage_bits[best_i]

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
