"""Covering design C(v,k,t) using a fast greedy algorithm with NumPy bitsets."""
import time
import itertools
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    DEADLINE = start_time + 110  # safety margin before 120s kill

    # Precompute t-subsets and map to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {c: i for i, c in enumerate(all_tsubs)}

    # Precompute for each point the list of t-subset indices it belongs to
    # point_masks[p] = bitmask (or list) of t-subsets containing p
    # For memory efficiency, store as list of indices
    point_tsubs = []
    for p in range(v):
        subs = []
        for c in all_tsubs:
            if p in c:
                subs.append(tsub_to_idx[c])
        point_tsubs.append(subs)

    # For each pair of points, precompute t-subsets containing both
    # This helps when building blocks incrementally
    # pair_tsubs[p1][p2] = list of t-subset indices containing both p1 and p2
    # But this is O(v^2 * C(v,t)) which is too much. Instead compute on the fly.

    # Active set: which t-subsets are still uncovered
    # Use a boolean numpy array for speed
    active = np.ones(n_tsubs, dtype=np.int8)
    active_count = n_tsubs

    blocks = []

    while active_count > 0:
        if time.time() > DEADLINE:
            break

        # Find the first uncovered t-subset (smallest index)
        # We could be smarter, but let's use the first one for determinism
        # Actually, let's pick the t-subset with the fewest candidate points
        # that can still cover it, to make the block more constrained.
        # But for speed, just pick the first uncovered one.
        uncovered_indices = np.where(active > 0)[0]
        if len(uncovered_indices) == 0:
            break
        # Pick the first uncovered t-subset
        first_idx = uncovered_indices[0]
        seed_tsub = all_tsubs[first_idx]
        block = list(seed_tsub)

        # Now grow the block to size k
        while len(block) < k:
            if time.time() > DEADLINE:
                break

            # Compute the set of uncovered t-subsets that can be covered
            # by adding a new point x to the current block.
            # A t-subset s is coverable by adding x iff:
            #   s contains x, and s \ {x} is a (t-1)-subset of the current block.
            # So for each (t-1)-subset r of block, and each x not in block,
            # the t-subset r + {x} is covered.

            # Precompute all (t-1)-subsets of the current block
            block_set = set(block)
            t_minus_1 = t - 1
            if t_minus_1 <= len(block):
                t1_subsets = list(itertools.combinations(block, t_minus_1))
            else:
                t1_subsets = []

            # For each candidate x not in block, count how many uncovered t-subsets it would cover
            # A t-subset (t1_sub + (x,)) is covered if it's in uncovered set
            # Build a dict: x -> count of newly covered t-subsets
            gains = {}
            for x in range(v):
                if x in block_set:
                    continue
                count = 0
                for r in t1_subsets:
                    # The t-subset is sorted(r + (x,))
                    s = tuple(sorted(r + (x,)))
                    if s in tsub_to_idx:
                        idx = tsub_to_idx[s]
                        if active[idx] > 0:
                            count += 1
                gains[x] = count

            if not gains:
                break

            # Pick the x with maximum gain; tie-break by smallest x
            best_x = min(gains.keys(), key=lambda x: (-gains[x], x))
            block.append(best_x)

        # Sort the block
        block.sort()
        blocks.append(tuple(block))

        # Update active set: mark all t-subsets in this block as covered
        for c in itertools.combinations(block, t):
            idx = tsub_to_idx[c]
            if active[idx] > 0:
                active[idx] = 0
                active_count -= 1

    return [list(b) for b in blocks]
