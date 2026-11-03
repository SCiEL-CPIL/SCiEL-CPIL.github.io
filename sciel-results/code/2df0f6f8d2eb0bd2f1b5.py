"""Vectorized greedy covering for C(19,8,5) with deadline guard."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np


def construct(v, k, t):
    deadline = time.time() + 110.0  # safety margin before 120s kill

    # Precompute all t-subsets as bitmasks
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_masks = np.array([sum(1 << x for x in c) for c in all_t], dtype=np.int64)

    # For each point x, precompute which t-subsets contain x (as bitmask indices)
    point_to_t = []
    for x in range(v):
        idxs = [i for i, c in enumerate(all_t) if x in c]
        point_to_t.append(np.array(idxs, dtype=np.int32))

    # Coverage: which t-subsets are covered so far
    covered = np.zeros(n_t, dtype=bool)
    uncovered_count = n_t

    # Count how many times each t-subset is covered (for dedup / block quality)
    cover_count = np.zeros(n_t, dtype=np.int32)

    blocks = []
    seen_blocks = set()

    while uncovered_count > 0:
        if time.time() > deadline:
            break

        # Choose seed: uncovered t-subset with minimum "potential" (fewest points that
        # could help), i.e., pick an uncovered subset. Use first uncovered for determinism,
        # but prefer ones that are "hardest" - actually just pick first uncovered.
        unc_idx = np.where(~covered)[0]
        if len(unc_idx) == 0:
            break
        seed_idx = unc_idx[0]
        block = list(all_t[seed_idx])  # start with t points from the uncovered subset

        # Greedily add points to maximize newly covered t-subsets
        for _ in range(k - t):
            if time.time() > deadline:
                break
            if len(block) >= k:
                break

            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Compute gain: how many currently-uncovered t-subsets contain x AND
                # have all other t-1 points in block
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    mask = sum(1 << y for y in rest) | (1 << x)
                    # Find which t-subsets have this exact mask
                    # We need t-subsets that are subsets of block ∪ {x} and contain x
                    # i.e., t-subsets S where x in S and S \ {x} subset of block
                    # These are exactly the t-subsets formed by choosing t-1 from block and adding x
                    for rest2 in itertools.combinations(block, t - 1):
                        pass
                # More efficient: precomputed approach
                # For point x, the t-subsets that would be covered by adding x to block are
                # those t-subsets S where x in S and S \ {x} ⊆ block
                # We can compute this by: for each (t-1)-subset R of block, S = R ∪ {x}
                # Check if S is in all_t and is uncovered
                for R in itertools.combinations(block, t - 1):
                    S = tuple(sorted(R + (x,)))
                    # Find index of S in all_t
                    # We need a fast lookup
                    smask = sum(1 << y for y in S)
                    # Binary search or dict lookup
                    if smask in mask_to_idx:
                        i = mask_to_idx[smask]
                        if not covered[i]:
                            gain += 1

                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)

        block.sort()
        bkey = tuple(block)
        if bkey in seen_blocks:
            # Try to find a distinct block; for now skip (shouldn't happen often)
            continue
        seen_blocks.add(bkey)
        blocks.append(block)

        # Update coverage
        for S in itertools.combinations(block, t):
            smask = sum(1 << y for y in S)
            if smask in mask_to_idx:
                i = mask_to_idx[smask]
                if not covered[i]:
                    covered[i] = True
                    uncovered_count -= 1
                cover_count[i] += 1

    # Verify completeness; if incomplete, fill with more greedy blocks
    if uncovered_count > 0 and time.time() < deadline:
        # Continue with a simpler loop to finish
        while uncovered_count > 0:
            if time.time() > deadline - 5:
                break
            unc_idx = np.where(~covered)[0]
            if len(unc_idx) == 0:
                break
            seed_idx = unc_idx[0]
            block = list(all_t[seed_idx])
            for _ in range(k - t):
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for R in itertools.combinations(block, t - 1):
                        S = tuple(sorted(R + (x,)))
                        smask = sum(1 << y for y in S)
                        if smask in mask_to_idx:
                            i = mask_to_idx[smask]
                            if not covered[i]:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
            block.sort()
            bkey = tuple(block)
            if bkey not in seen_blocks:
                seen_blocks.add(bkey)
                blocks.append(block)
            for S in itertools.combinations(block, t):
                smask = sum(1 << y for y in S)
                if smask in mask_to_idx:
                    i = mask_to_idx[smask]
                    if not covered[i]:
                        covered[i] = True
                        uncovered_count -= 1

    return blocks


# Precompute mask_to_idx for fast lookup (module-level, computed once)
# But we need it inside construct... let's restructure to compute it lazily
_mask_to_idx_cache = {}

def _get_mask_to_idx(v, t):
    key = (v, t)
    if key in _mask_to_idx_cache:
        return _mask_to_idx_cache[key]
    all_t = list(itertools.combinations(range(v), t))
    m2i = {}
    for i, c in enumerate(all_t):
        m = sum(1 << x for x in c)
        m2i[m] = i
    _mask_to_idx_cache[key] = m2i
    return m2i

# Patch construct to use the cached lookup
def construct(v, k, t):
    deadline = time.time() + 110.0

    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    mask_to_idx = _get_mask_to_idx(v, t)

    covered = np.zeros(n_t, dtype=bool)
    uncovered_count = n_t
    blocks = []
    seen_blocks = set()

    while uncovered_count > 0:
        if time.time() > deadline:
            break

        unc_idx = np.where(~covered)[0]
        if len(unc_idx) == 0:
            break
        seed_idx = unc_idx[0]
        block = list(all_t[seed_idx])

        for _ in range(k - t):
            if time.time() > deadline:
                break
            if len(block) >= k:
                break
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for R in itertools.combinations(block, t - 1):
                    S = tuple(sorted(R + (x,)))
                    smask = sum(1 << y for y in S)
                    i = mask_to_idx.get(smask)
                    if i is not None and not covered[i]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)

        block.sort()
        bkey = tuple(block)
        if bkey in seen_blocks:
            continue
        seen_blocks.add(bkey)
        blocks.append(block)

        for S in itertools.combinations(block, t):
            smask = sum(1 << y for y in S)
            i = mask_to_idx.get(smask)
            if i is not None and not covered[i]:
                covered[i] = True
                uncovered_count -= 1

    return blocks
# EVOLVE-BLOCK-END
