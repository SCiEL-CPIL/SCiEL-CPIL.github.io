"""Improved greedy covering design C(19,7,4) with vectorized gains and local search."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 115.0  # safety margin under 120s
    all_ts = list(itertools.combinations(range(v), t))
    t_to_idx = {c: i for i, c in enumerate(all_ts)}
    n_t = len(all_ts)

    # Precompute, for each point x, the bitmask of t-subsets containing x (as a set of indices)
    # For gain computation we need, for a (t-1)-subset S and candidate x, the t-subset S∪{x}.
    # Represent uncovered as a boolean numpy array of length n_t.
    uncovered = np.ones(n_t, dtype=np.int32)
    covered_count = 0

    # Precompute for each (t-1) combination a list of (candidate x, t_idx) where x not in combo
    # Actually we compute gain(x) = number of uncovered t-subsets of form combo ∪ {x} for combo in C(block, t-1)
    # We'll build a map: for a given (t-1) tuple combo, the t-subsets containing it are combo ∪ {x} for x not in combo.
    # Precompute: for each (t-1) combo, list of (x, t_idx)
    t_minus_1_combos = list(itertools.combinations(range(v), t - 1))
    combo_candidates = {}
    for combo in t_minus_1_combos:
        combo_set = set(combo)
        lst = []
        for x in range(v):
            if x in combo_set:
                continue
            tc = tuple(sorted(combo + (x,)))
            lst.append((x, t_to_idx[tc]))
        combo_candidates[combo] = lst

    blocks = []
    block_set = set()

    def gain_for_block(block, uncovered):
        """Compute gain for each candidate x not in block."""
        block_set_local = set(block)
        # For each (t-1) combo of block, for each (x, t_idx), if x not in block, add to gain[x]
        gain = np.zeros(v, dtype=np.int64)
        for combo in itertools.combinations(block, t - 1):
            for x, t_idx in combo_candidates[combo]:
                if x in block_set_local:
                    continue
                if uncovered[t_idx]:
                    gain[x] += 1
        return gain

    while True:
        if time.time() > deadline:
            break
        if not uncovered.any():
            break

        # Pick the t-subset with highest "hardness" — just pick first uncovered
        # Better: pick uncovered t-subset that is in the fewest possible blocks? For greedy, pick first.
        # Use the first uncovered index for determinism but try to pick a "central" one.
        unc_idx = int(np.argmax(uncovered))  # first True
        start_combo = all_ts[unc_idx]
        block = list(start_combo)

        # Greedily fill the block
        while len(block) < k:
            if time.time() > deadline:
                break
            gain = gain_for_block(block, uncovered)
            # Mask out points already in block
            for x in block:
                gain[x] = -1
            best_x = int(np.argmax(gain))
            if gain[best_x] <= 0:
                # No gain; pick any unused point
                used = set(block)
                best_x = next(x for x in range(v) if x not in used)
            block.append(best_x)

        block.sort()
        bkey = tuple(block)
        if bkey in block_set:
            # Duplicates shouldn't happen with greedy, but guard
            pass
        block_set.add(bkey)
        blocks.append(block)

        # Update uncovered
        for tc in itertools.combinations(block, t):
            uncovered[t_to_idx[tc]] = 0

    # If we timed out mid-block, we may have an incomplete last block; ensure validity by finishing
    # Actually if we broke mid-block, the block isn't appended, and uncovered may be non-empty.
    # We must return a VALID covering. So if uncovered remains, we need to finish covering.
    # To be safe, if we exited due to time, do a fast fallback to cover remaining.
    if uncovered.any() and time.time() < deadline - 2.0:
        # Fast greedy to finish
        while uncovered.any() and time.time() < deadline - 1.0:
            unc_idx = int(np.argmax(uncovered))
            start_combo = all_ts[unc_idx]
            block = list(start_combo)
            while len(block) < k:
                gain = gain_for_block(block, uncovered)
                for x in block:
                    gain[x] = -1
                best_x = int(np.argmax(gain))
                if gain[best_x] <= 0:
                    used = set(block)
                    best_x = next(x for x in range(v) if x not in used)
                block.append(best_x)
            block.sort()
            bkey = tuple(block)
            if bkey not in block_set:
                block_set.add(bkey)
                blocks.append(block)
                for tc in itertools.combinations(block, t):
                    uncovered[t_to_idx[tc]] = 0

    # Local search: try to remove redundant blocks
    def covers_all(blocks, uncovered):
        """Return uncovered array for a given list of blocks."""
        uc = np.zeros(n_t, dtype=np.int32)
        for b in blocks:
            for tc in itertools.combinations(b, t):
                uc[t_to_idx[tc]] = 1
        return 1 - uc  # 1 where not covered

    # Try removing each block if the rest still covers everything
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        n = len(blocks)
        for i in range(n):
            if time.time() > deadline - 5.0:
                break
            # Check if block i is redundant
            # Compute coverage without block i
            # Quick check: is there a t-subset in block i that is ONLY in block i?
            # For each t-subset of block i, count how many blocks cover it.
            redundant = True
            for tc in itertools.combinations(blocks[i], t):
                idx = t_to_idx[tc]
                # Count blocks other than i that contain tc
                found = False
                for j, b in enumerate(blocks):
                    if j == i:
                        continue
                    if set(tc).issubset(set(b)):
                        found = True
                        break
                if not found:
                    redundant = False
                    break
            if redundant:
                blocks.pop(i)
                improved = True
                break

    # Try merging: replace two blocks with one? Not straightforward for k=7.
    # Try swapping points in a block to cover more uncovered / reduce others.
    # Local search: for each block, try replacing one point with another to increase coverage of "rare" t-subsets.

    # Final: ensure no duplicates and valid
    seen = set()
    final_blocks = []
    for b in blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            final_blocks.append(list(b))

    return final_blocks
# EVOLVE-BLOCK-END
