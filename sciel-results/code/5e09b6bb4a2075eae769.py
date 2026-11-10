"""Bitset-accelerated greedy covering for C(24,7,3)."""
import time
import itertools
import random
from math import comb


def construct(v, k, t):
    # Deadline: evaluator kills at 150s wall-clock; use 140s as safety margin.
    deadline = time.time() + 140.0

    # Precompute all t-subsets and assign each an index.
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)  # C(24,3) = 2024
    tsub_index = {sub: i for i, sub in enumerate(all_tsubs)}

    # Precompute coverage mask for every k-subset: a Python int bitmask.
    # bit i is 1 iff tsub_index[sub] == i is covered by this k-subset.
    # C(24,7) = 346104 k-subsets, each covers C(7,3)=35 t-subsets.
    ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(ksubs)

    # coverage_mask[ks] : int with bits set for all t-subsets contained in ksubs[ks]
    coverage_mask = [0] * n_ksubs
    for ks_idx, ks in enumerate(ksubs):
        mask = 0
        for ts in itertools.combinations(ks, t):
            mask |= (1 << tsub_index[ts])
        coverage_mask[ks_idx] = mask

    # uncovered: bitmask of all t-subsets (initially all 1s)
    full_mask = (1 << n_tsubs) - 1
    uncovered = full_mask

    blocks = []
    rng = random.Random(42)

    # Greedy: repeatedly pick the k-subset that covers the most uncovered t-subsets.
    # To speed up, we can use a multi-pass approach:
    #   Pass 1: full scan over all k-subsets to find best.
    #   This is O(n_ksubs * cost_of_bit_count). Python int.bit_count() is fast.
    #
    # For C(24,7,3): n_ksubs=346104, n_tsubs=2024.
    # Each iteration: 346104 bit-ANDs + bit_counts. Should be ~0.1-0.3s per block.
    # We need ~76 blocks -> ~10-25s total. Well within budget.

    while uncovered and time.time() < deadline - 5.0:
        # Find k-subset with maximum uncovered coverage
        best_gain = -1
        best_idx = -1
        # Use local variables for speed
        cm = coverage_mask
        unc = uncovered
        for i in range(n_ksubs):
            gain = (cm[i] & unc).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                if gain == 0:
                    break  # nothing more to cover
        if best_gain <= 0:
            break

        block = list(ksubs[best_idx])
        blocks.append(block)
        uncovered &= ~coverage_mask[best_idx]

    # If we ran out of time or got stuck, try a randomized restart with partial coverage
    # to potentially find a better arrangement.
    if uncovered and time.time() < deadline - 10.0:
        # Try a few randomized greedy restarts from scratch
        best_blocks = blocks
        best_count = len(blocks)

        for restart in range(5):
            if time.time() > deadline - 10.0:
                break
            unc = full_mask
            blks = []
            # Shuffle the order of k-subsets to break symmetry
            indices = list(range(n_ksubs))
            rng.shuffle(indices)

            while unc and time.time() < deadline - 10.0:
                best_gain = -1
                best_idx = -1
                for i in indices:
                    gain = (coverage_mask[i] & unc).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                        if gain == 0:
                            break
                if best_gain <= 0:
                    break
                blks.append(list(ksubs[best_idx]))
                unc &= ~coverage_mask[best_idx]

            if len(blks) < best_count:
                best_blocks = blks
                best_count = len(blks)

        blocks = best_blocks

    # Final: if still uncovered (shouldn't happen for valid covering), do a
    # last-ditch exact fill for remaining t-subsets (only if time permits).
    if uncovered and time.time() < deadline - 2.0:
        # For each uncovered t-subset, add a block containing it
        # This is a fallback; the greedy should have covered everything.
        remaining = []
        for ts in all_tsubs:
            if (uncovered >> tsub_index[ts]) & 1:
                remaining.append(ts)
        for ts in remaining:
            # Find any k-subset containing ts that covers the most remaining
            best_gain = -1
            best_idx = -1
            for i in range(n_ksubs):
                ks = ksubs[i]
                if all(x in ks for x in ts):
                    gain = (coverage_mask[i] & uncovered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
            if best_idx >= 0:
                blocks.append(list(ksubs[best_idx]))
                uncovered &= ~coverage_mask[best_idx]

    return blocks
