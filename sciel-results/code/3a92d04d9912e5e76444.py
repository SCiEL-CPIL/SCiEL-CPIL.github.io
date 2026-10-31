"""Covering design C(19,7,4): minimize number of 7-subsets covering all 4-subsets.
Strategy: fast vectorized greedy + aggressive local search (removal, swap, 2-swap)
with multiple restarts and a final polish phase.
"""
import time
import itertools
import random
import numpy as np

_DEADLINE = 145.0
_START = None


def _elapsed():
    return time.time() - _START


def _precompute(v, k, t):
    """Precompute all t-subsets and for every k-subset its coverage as a NumPy uint8 array."""
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_index = {c: i for i, c in enumerate(all_t)}
    k_subsets = list(itertools.combinations(range(v), k))
    n_ks = len(k_subsets)
    coverage = np.zeros((n_ks, n_t), dtype=np.uint8)
    for i, ks in enumerate(k_subsets):
        for ts in itertools.combinations(ks, t):
            coverage[i, t_index[ts]] = 1
    return all_t, t_index, k_subsets, coverage


def _greedy_construct(coverage, rng, top_k=1):
    """Vectorized greedy set cover with optional top-k randomization."""
    n_ks, n_t = coverage.shape
    uncovered = np.ones(n_t, dtype=np.uint8)
    blocks = []
    while uncovered.any() and _elapsed() < _DEADLINE - 5:
        gains = coverage.dot(uncovered)
        if gains.max() == 0:
            break
        if top_k > 1:
            sorted_idx = np.argsort(gains)[::-1]
            best_i = sorted_idx[rng.randrange(min(top_k, n_ks))]
        else:
            best_i = int(gains.argmax())
        blocks.append(best_i)
        uncovered[coverage[best_i] > 0] = 0
    return blocks


def _is_fully_covered(blocks, coverage):
    """Check if blocks cover all t-subsets."""
    if not blocks:
        return False
    union = np.zeros(coverage.shape[1], dtype=np.uint8)
    for b in blocks:
        union |= coverage[b]
    return union.all()


def _local_search(blocks, coverage, rng, time_budget):
    """Local search: fast removal via coverage counts, swap, and pair-swap moves."""
    blocks = list(blocks)
    n_ks, n_t = coverage.shape

    # Precompute for each k-subset which t-subsets it covers (row of coverage matrix)
    # We'll maintain a count array: count[i] = how many selected blocks cover t-subset i
    count = np.zeros(n_t, dtype=np.int32)
    for b in blocks:
        count += coverage[b]

    def is_fully_covered():
        return count.min() >= 1

    def try_remove_all():
        """Repeatedly remove any block that is redundant. Returns number removed."""
        nonlocal blocks, count
        removed = 0
        changed = True
        while changed and _elapsed() < time_budget:
            changed = False
            i = 0
            while i < len(blocks) and _elapsed() < time_budget:
                # Check if removing blocks[i] keeps full coverage
                c = coverage[blocks[i]]
                new_count = count - c
                if new_count.min() >= 0 and (new_count == 0).all() == False:
                    # Actually need: (count - c) >= 1 everywhere
                    if (count - c).min() >= 1:
                        count -= c
                        blocks.pop(i)
                        removed += 1
                        changed = True
                        continue
                i += 1
        return removed

    def swap_and_remove():
        """Try random swap: replace one block with another, then remove redundancies."""
        nonlocal blocks, count
        if len(blocks) < 2:
            return False
        i = rng.randrange(len(blocks))
        j = rng.randrange(n_ks)
        if j in blocks:
            return False
        old = blocks[i]
        # Apply swap
        count -= coverage[old]
        count += coverage[j]
        blocks[i] = j
        # Try removals
        n_removed = try_remove_all()
        if n_removed > 0:
            return True
        else:
            # Revert
            count -= coverage[j]
            count += coverage[old]
            blocks[i] = old
            return False

    def pair_swap_and_remove():
        """Replace two blocks with two new ones, then remove redundancies."""
        nonlocal blocks, count
        if len(blocks) < 3:
            return False
        i1 = rng.randrange(len(blocks))
        i2 = rng.randrange(len(blocks))
        if i1 == i2:
            return False
        j1 = rng.randrange(n_ks)
        j2 = rng.randrange(n_ks)
        if j1 in blocks or j2 in blocks or j1 == j2:
            return False
        old1, old2 = blocks[i1], blocks[i2]
        # Apply
        count -= coverage[old1]
        count -= coverage[old2]
        count += coverage[j1]
        count += coverage[j2]
        blocks[i1] = j1
        blocks[i2] = j2
        # Try removals
        n_removed = try_remove_all()
        if n_removed > 0:
            return True
        else:
            # Revert
            count -= coverage[j1]
            count -= coverage[j2]
            count += coverage[old1]
            count += coverage[old2]
            blocks[i1] = old1
            blocks[i2] = old2
            return False

    improved = True
    while improved and _elapsed() < time_budget:
        improved = False

        # --- Removal pass ---
        n_rem = try_remove_all()
        if n_rem > 0:
            improved = True

        # --- Swap pass ---
        n_swap_tries = 60
        for _ in range(n_swap_tries):
            if _elapsed() >= time_budget:
                break
            if swap_and_remove():
                improved = True

        # --- Pair-swap pass ---
        if len(blocks) >= 3 and _elapsed() < time_budget:
            n_pair_tries = 20
            for _ in range(n_pair_tries):
                if _elapsed() >= time_budget:
                    break
                if pair_swap_and_remove():
                    improved = True

        # --- Repair if needed ---
        if not is_fully_covered():
            uncovered_mask = count == 0
            gains = coverage.dot(uncovered_mask.astype(np.uint8))
            if gains.max() > 0:
                best_i = int(gains.argmax())
                blocks.append(best_i)
                count += coverage[best_i]

    return blocks


def construct(v, k, t):
    global _START
    _START = time.time()
    rng = random.Random(12345)

    all_t, t_index, k_subsets, coverage = _precompute(v, k, t)
    n_t = len(all_t)

    best_blocks = None
    best_len = float('inf')

    # Multiple restarts with varying top_k
    restarts = [
        (1, 1),
        (3, 2),
        (5, 3),
        (7, 5),
        (10, 7),
    ]

    ls_time = 40.0
    time_per_restart = max(5.0, (_DEADLINE - ls_time - 5) / len(restarts))

    for top_k, seed_offset in restarts:
        if _elapsed() > _DEADLINE - ls_time - 5:
            break
        rng_r = random.Random(12345 + seed_offset)
        t_start = _elapsed()
        blocks = _greedy_construct(coverage, rng_r, top_k=top_k)
        blocks = _local_search(blocks, coverage, rng_r, t_start + time_per_restart)
        if _is_fully_covered(blocks, coverage) and len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)

    # Final local search on best solution with remaining time
    if best_blocks is not None:
        remaining = _DEADLINE - 3 - _elapsed()
        if remaining > 2:
            best_blocks = _local_search(best_blocks, coverage, rng, _elapsed() + remaining)
        # Extra polish: another pass with a fresh RNG
        remaining2 = _DEADLINE - 2 - _elapsed()
        if remaining2 > 2:
            rng2 = random.Random(99999)
            best_blocks = _local_search(best_blocks, coverage, rng2, _elapsed() + remaining2)

    # Safety: ensure valid
    if best_blocks is None or not _is_fully_covered(best_blocks, coverage):
        blocks = _greedy_construct(coverage, rng, top_k=1)
        if not _is_fully_covered(blocks, coverage):
            uncovered = np.ones(n_t, dtype=np.uint8)
            for b in blocks:
                uncovered[coverage[b] > 0] = 0
            while uncovered.any() and _elapsed() < _DEADLINE - 1:
                gains = coverage.dot(uncovered)
                if gains.max() == 0:
                    break
                best_i = int(gains.argmax())
                blocks.append(best_i)
                uncovered[coverage[best_i] > 0] = 0
        best_blocks = blocks

    result = [list(k_subsets[b]) for b in best_blocks]
    seen = set()
    dedup = []
    for blk in result:
        key = tuple(sorted(blk))
        if key not in seen:
            seen.add(key)
            dedup.append(list(key))
    return dedup
