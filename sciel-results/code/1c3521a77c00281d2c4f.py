"""Stronger covering design: bitmask greedy + iterative local search (block removal)."""
import time
import itertools
import random
from collections import defaultdict

import numpy as np


def construct(v, k, t):
    start = time.time()
    deadline = start + 110  # safety margin before 120s kill

    # Precompute all t-subsets as bitmasks and index them
    # For v=19, k=8, t=5: C(19,5)=11628 subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)

    # Map each t-subset to an index
    t_subset_to_idx = {tuple(cs): i for i, cs in enumerate(all_t_subsets)}

    # Convert t-subsets to bitmasks for fast membership
    t_masks = []
    for cs in all_t_subsets:
        mask = 0
        for p in cs:
            mask |= (1 << p)
        t_masks.append(mask)
    t_masks = np.array(t_masks, dtype=np.int64)

    # For each block (k-subset), precompute which t-subsets it covers
    # C(8,5) = 56 t-subsets per block
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subsets)

    # For each k-subset, store the set of t-subset indices it covers
    # and its bitmask
    k_to_t_indices = []
    k_masks = []
    for ks in all_k_subsets:
        kmask = 0
        for p in ks:
            kmask |= (1 << p)
        k_masks.append(kmask)
        # Find all t-subsets contained in this k-subset
        idxs = []
        for tc in itertools.combinations(ks, t):
            idxs.append(t_subset_to_idx[tuple(tc)])
        k_to_t_indices.append(idxs)

    k_masks = np.array(k_masks, dtype=np.int64)

    # Greedy set cover using bitmasks
    # uncovered: boolean array of length n_t
    uncovered = np.ones(n_t, dtype=np.bool_)
    blocks = []  # list of k-subset indices chosen

    # For speed, precompute for each point and each t-subset whether point is in it
    # point_in_t[point] = bitmask of t-subset indices containing that point
    point_in_t = np.zeros((v, n_t), dtype=np.bool_)
    for i, tc in enumerate(all_t_subsets):
        for p in tc:
            point_in_t[p, i] = True

    def greedy_cover():
        nonlocal uncovered, blocks
        uncovered = np.ones(n_t, dtype=np.bool_)
        blocks = []

        while uncovered.any():
            if time.time() > deadline:
                break

            # Find the first uncovered t-subset to anchor the block
            first_idx = np.argmax(uncovered)
            anchor_mask = t_masks[first_idx]
            anchor_points = [p for p in range(v) if (anchor_mask >> p) & 1]

            # Build block greedily: start with one point from anchor, add points maximizing gain
            # Try multiple starting points from anchor for better quality
            best_block = None
            best_gain_total = -1

            for start_p in anchor_points:
                if time.time() > deadline:
                    break
                block = [start_p]
                block_mask = 1 << start_p
                # Count t-subsets covered so far
                # A t-subset is covered if all its points are in block
                # For partial block, count t-subsets that are subsets of block
                # Actually for gain we need: t-subsets that contain all current block points
                # and will be fully covered when we add the new point

                # Simpler: at each step, for each candidate point x not in block,
                # gain = number of uncovered t-subsets that are subsets of (block + {x})
                # A t-subset S is subset of block+{x} iff S is subset of block or (S\{x} is subset of block and x in S)
                # More precisely: S is subset of (block ∪ {x}) iff (S ∩ {x}) ⊆ {x} and S ⊆ block ∪ {x}
                # i.e., S \ {x} ⊆ block and S ∩ {x} ⊆ {x} (always true if x in S or not)
                # So: S is subset of block∪{x} iff (S \ {x}) ⊆ block
                # This means: all points of S except possibly x are in block

                # For efficiency, precompute for each point x and current block_mask:
                # count of uncovered t-subsets S where (S \ {x}) ⊆ block
                # This is equivalent to: S ⊆ block ∪ {x} and x ∈ S or x ∉ S
                # Actually: S ⊆ block ∪ {x} iff S ∩ complement(block ∪ {x}) = ∅
                # Let's use a different approach: for each candidate x, 
                # the t-subsets newly covered are those S where S ⊆ block∪{x} but S ⊄ block
                # i.e., x ∈ S and S\{x} ⊆ block

                # Precompute: for each point x, for each subset of points (as mask), 
                # the count of uncovered t-subsets S where x ∈ S and S\{x} ⊆ mask
                # This is hard to precompute for all masks. Instead, compute on the fly.

                # Alternative: for each candidate x, compute gain by checking each uncovered t-subset
                # But that's O(n_t * v) per step which is 11628 * 19 ≈ 220k per step, and we have ~335*8 = 2680 steps
                # That's ~590M operations... too slow in pure Python.

                # Better: use the precomputed point_in_t array.
                # For a candidate x and current block_mask:
                # gain(x) = number of uncovered t-subsets S such that S ⊆ block ∪ {x} and x ∈ S
                # S ⊆ block ∪ {x} and x ∈ S iff S\{x} ⊆ block
                # S\{x} ⊆ block means: for all p in S, if p ≠ x then p in block
                # Equivalently: S ∩ complement(block) ⊆ {x} and x ∈ S
                # i.e., (S & ~block_mask) has at most one bit, and that bit is x, and x ∈ S

                # Let's compute: for each uncovered t-subset index i, check if t_masks[i] & ~block_mask is either 0 or (1<<x)
                # If it's (1<<x), then adding x covers it.

                # Vectorized: for all uncovered t-subsets, compute mask_diff = t_masks[uncovered] & ~block_mask
                # For candidate x: gain = count of mask_diff == (1 << x)
                # We can do this with numpy.

                current_uncovered_idx = np.where(uncovered)[0]
                if len(current_uncovered_idx) == 0:
                    break

                # For each candidate point not in block:
                candidates = [p for p in range(v) if p not in block]
                best_x = None
                best_g = -1
                for x in candidates:
                    x_mask = 1 << x
                    # t-subsets S where S\{x} ⊆ block means S & ~block_mask == x_mask (and x in S)
                    # Actually S & ~block_mask could be 0 (S already subset of block, but then it's already covered)
                    # We want S where x ∈ S and S\{x} ⊆ block
                    # S & ~block_mask: bits of S not in block. For S\{x} ⊆ block, we need S & ~block_mask ⊆ {x}
                    # So S & ~block_mask is either 0 or (1<<x)
                    # If 0: S ⊆ block, already covered (shouldn't be in uncovered)
                    # If (1<<x): x is the only point of S not in block, so adding x covers S
                    diff = t_masks[current_uncovered_idx] & ~block_mask
                    # Count where diff == x_mask
                    g = np.sum(diff == x_mask)
                    if g > best_g:
                        best_g = g
                        best_x = x

                if best_x is None or best_g <= 0:
                    # No point improves coverage; pick any remaining
                    if candidates:
                        best_x = candidates[0]
                    else:
                        break

                block.append(best_x)
                block_mask |= (1 << best_x)

                # Update uncovered: remove t-subsets that are now subsets of block
                # S ⊆ block iff S & ~block_mask == 0
                diff_all = t_masks & ~block_mask
                newly_covered = (diff_all == 0)
                uncovered[uncovered & newly_covered] = False

                if len(block) == k:
                    break

            # Pick the best starting point's block
            if best_block is None:
                # Use the last computed block
                pass

            # Actually, let me simplify: just do single-path greedy from first anchor point
            # The multi-start is complex. Let's do a simpler but effective greedy.
            pass

        return blocks

    # Simpler and faster greedy:
    def greedy_cover_v2():
        nonlocal uncovered
        uncovered = np.ones(n_t, dtype=np.bool_)
        chosen_blocks = []

        while uncovered.any():
            if time.time() > deadline:
                break

            # Pick first uncovered t-subset
            first_idx = int(np.argmax(uncovered))
            anchor = list(all_t_subsets[first_idx])

            # Build block starting from anchor
            block = [anchor[0]]
            block_set = set(block)

            while len(block) < k:
                if time.time() > deadline:
                    break
                block_mask = 0
                for p in block:
                    block_mask |= (1 << p)

                best_x = None
                best_g = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    x_mask = 1 << x
                    # Count uncovered t-subsets S where S\{x} ⊆ block
                    # i.e., S & ~block_mask == x_mask (x is the only point not in block)
                    # OR S & ~block_mask == 0 (but those should already be covered)
                    # We want S where adding x makes S ⊆ block∪{x}
                    # S ⊆ block∪{x} iff S & ~(block_mask | x_mask) == 0
                    # And x ∈ S (otherwise S was already ⊆ block)
                    # So: (S & ~(block_mask | x_mask)) == 0 and (S & x_mask) != 0
                    # Which is: (S & ~block_mask) == x_mask

                    diff = t_masks & ~block_mask
                    # We need diff == x_mask for the t-subset to be newly covered by adding x
                    # But we only count uncovered ones
                    mask = uncovered & (diff == x_mask)
                    g = np.sum(mask)
                    if g > best_g:
                        best_g = g
                        best_x = x

                if best_x is None:
                    # Pick arbitrary
                    for x in range(v):
                        if x not in block_set:
                            best_x = x
                            break
                    if best_x is None:
                        break

                block.append(best_x)
                block_set.add(best_x)

                # Update uncovered
                block_mask = 0
                for p in block:
                    block_mask |= (1 << p)
                diff_all = t_masks & ~block_mask
                newly_covered = (diff_all == 0)
                uncovered[uncovered & newly_covered] = False

            # Convert block to k-subset index
            block.sort()
            # Find index in all_k_subsets
            block_tuple = tuple(block)
            # Binary search or linear
            # For speed, use a dict
            if not hasattr(greedy_cover_v2, 'k_idx_map'):
                greedy_cover_v2.k_idx_map = {tuple(ks): i for i, ks in enumerate(all_k_subsets)}
            k_idx = greedy_cover_v2.k_idx_map.get(block_tuple)
            if k_idx is not None:
                chosen_blocks.append(k_idx)

        return chosen_blocks

    # Run greedy
    blocks = greedy_cover_v2()

    # Local search: iteratively try to remove blocks
    # Build coverage count: for each t-subset, how many blocks cover it
    coverage_count = np.zeros(n_t, dtype=np.int32)
    for k_idx in blocks:
        for t_idx in k_to_t_indices[k_idx]:
            coverage_count[t_idx] += 1

    # Iterative removal
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each block
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline:
                break
            k_idx = blocks[i]
            # Check if removing this block leaves all t-subsets covered
            # A t-subset is at risk if coverage_count[t_idx] == 1 and it's covered by block i
            # Check all t-subsets covered by block i
            safe = True
            for t_idx in k_to_t_indices[k_idx]:
                if coverage_count[t_idx] <= 1:
                    safe = False
                    break
            if safe:
                # Remove block
                for t_idx in k_to_t_indices[k_idx]:
                    coverage_count[t_idx] -= 1
                blocks.pop(i)
                improved = True
                # Restart scan from beginning
                break

    # Second pass: try swapping blocks (replace one with another that covers more)
    # For now, just do removal passes

    # Convert back to list of lists
    result = [list(all_k_subsets[k_idx]) for k_idx in blocks]

    # Verify and return
    return result
