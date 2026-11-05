"""Covering design C(v,k,t) via bitmask greedy + precomputed index local search."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def time_ok():
        return time.time() < deadline

    all_points = list(range(v))
    all_t_subs = list(itertools.combinations(all_points, t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Precompute all k-subsets and their covered t-subset indices
    all_k_subs = list(itertools.combinations(all_points, k))
    n_k = len(all_k_subs)

    k_sub_t_idxs = []
    for k_sub in all_k_subs:
        t_idxs = [t_sub_to_idx[ts] for ts in itertools.combinations(k_sub, t)]
        k_sub_t_idxs.append(t_idxs)

    # Reverse mapping: for each t-subset, which k-subsets cover it
    t_sub_to_k_idxs = [[] for _ in range(n_t)]
    for i, t_idxs in enumerate(k_sub_t_idxs):
        for t_idx in t_idxs:
            t_sub_to_k_idxs[t_idx].append(i)

    # Greedy Phase: anchor-based
    covered = np.zeros(n_t, dtype=np.uint8)
    uncovered_count = n_t
    blocks = []

    while uncovered_count > 0 and time_ok():
        uncovered_indices = np.where(covered == 0)[0]
        if len(uncovered_indices) == 0:
            break

        anchor_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
        anchor_t_sub = all_t_subs[anchor_idx]
        anchor_set = set(anchor_t_sub)

        remaining_points = [p for p in all_points if p not in anchor_set]
        k_minus_t = k - t

        best_block = None
        best_gain = -1

        for ext in itertools.combinations(remaining_points, k_minus_t):
            block = tuple(sorted(list(anchor_t_sub) + list(ext)))
            # Use precomputed lookup: find index of this block in all_k_subs
            # (blocks are sorted tuples, all_k_subs are sorted tuples)
            # We need a fast way to get the block's t-subset indices.
            # Compute gain directly via t_sub_to_idx.
            gain = 0
            for t_sub in itertools.combinations(block, t):
                idx = t_sub_to_idx[t_sub]
                if covered[idx] == 0:
                    gain += 1

            if gain > best_gain:
                best_gain = gain
                best_block = block

        if best_block is None:
            break

        blocks.append(best_block)

        # Update coverage
        for t_sub in itertools.combinations(best_block, t):
            idx = t_sub_to_idx[t_sub]
            if covered[idx] == 0:
                covered[idx] = 1
                uncovered_count -= 1

    # Convert blocks to indices into all_k_subs for local search
    k_sub_to_idx = {k_sub: i for i, k_sub in enumerate(all_k_subs)}
    block_idxs = [k_sub_to_idx[b] for b in blocks]

    # Reduction Phase: iteratively remove redundant blocks
    if block_idxs and time_ok():
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in block_idxs:
            for t_idx in k_sub_t_idxs[b_idx]:
                cover_count[t_idx] += 1

        changed = True
        while changed and time_ok():
            changed = False
            new_block_idxs = []
            for b_idx in block_idxs:
                keep = False
                for t_idx in k_sub_t_idxs[b_idx]:
                    if cover_count[t_idx] == 1:
                        keep = True
                        break
                if keep:
                    new_block_idxs.append(b_idx)
                else:
                    changed = True
                    for t_idx in k_sub_t_idxs[b_idx]:
                        cover_count[t_idx] -= 1
            block_idxs = new_block_idxs

        # Simulated Annealing local search
        if time_ok() and len(block_idxs) > 1:
            # Recompute cover counts
            cover_count = np.zeros(n_t, dtype=np.int32)
            for b_idx in block_idxs:
                for t_idx in k_sub_t_idxs[b_idx]:
                    cover_count[t_idx] += 1

            current_blocks = list(block_idxs)
            current_energy = len(current_blocks) + 1000.0 * int(np.sum(cover_count == 0))
            best_blocks = list(current_blocks)
            best_energy = current_energy

            T = 50.0
            cooling_rate = 0.995
            min_T = 0.05
            penalty = 1000.0
            rng = random.Random(42)

            while time.time() < deadline - 5.0 and T > min_T:
                # Move 1: Try to remove a random block
                if len(current_blocks) > 0:
                    i = rng.randint(0, len(current_blocks) - 1)
                    b_idx = current_blocks[i]

                    new_uncovered = 0
                    for t_idx in k_sub_t_idxs[b_idx]:
                        if cover_count[t_idx] == 1:
                            new_uncovered += 1

                    if new_uncovered == 0:
                        # Valid removal, always accept
                        for t_idx in k_sub_t_idxs[b_idx]:
                            cover_count[t_idx] -= 1
                        current_blocks.pop(i)
                        current_energy -= 1.0
                    else:
                        delta_E = 1.0 + penalty * new_uncovered
                        if rng.random() < np.exp(-delta_E / T):
                            for t_idx in k_sub_t_idxs[b_idx]:
                                cover_count[t_idx] -= 1
                            current_blocks.pop(i)
                            current_energy += delta_E

                # Move 2: Try to swap a block with another
                if len(current_blocks) > 1:
                    i = rng.randint(0, len(current_blocks) - 1)
                    b_idx = current_blocks[i]
                    # Pick a random other block to swap with
                    j = rng.randint(0, len(current_blocks) - 1)
                    if j == i:
                        j = (j + 1) % len(current_blocks)
                    b_idx2 = current_blocks[j]

                    # Compute effect of removing b_idx and adding b_idx2
                    # First remove b_idx
                    for t_idx in k_sub_t_idxs[b_idx]:
                        cover_count[t_idx] -= 1
                    # Then add b_idx2
                    for t_idx in k_sub_t_idxs[b_idx2]:
                        cover_count[t_idx] += 1

                    new_uncovered = int(np.sum(cover_count == 0))
                    delta_E = penalty * new_uncovered  # size unchanged

                    if delta_E <= 0 or rng.random() < np.exp(-delta_E / T):
                        current_blocks[i] = b_idx2
                        current_energy += delta_E
                    else:
                        # Revert
                        for t_idx in k_sub_t_idxs[b_idx2]:
                            cover_count[t_idx] -= 1
                        for t_idx in k_sub_t_idxs[b_idx]:
                            cover_count[t_idx] += 1

                if current_energy < best_energy:
                    best_energy = current_energy
                    best_blocks = list(current_blocks)

                T *= cooling_rate

            # Use best found
            block_idxs = best_blocks

    # Convert back to list of blocks
    return [list(all_k_subs[b_idx]) for b_idx in block_idxs]
# EVOLVE-BLOCK-END
