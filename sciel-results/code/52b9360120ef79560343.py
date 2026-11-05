import itertools
import random
import time
import math
import numpy as np

# Global Precomputation
V = 19
K = 7
T = 4

# Map 4-subsets to indices
subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

# Generate all 7-subsets and their masks
all_blocks = []
all_masks = []
blocks_containing_subset = [[] for _ in range(N_SUBSETS)]
subset_frequencies = [0] * N_SUBSETS

for block in itertools.combinations(range(V), K):
    mask = 0
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        mask |= 1 << idx
        blocks_containing_subset[idx].append(len(all_blocks))
        subset_frequencies[idx] += 1
    all_blocks.append(block)
    all_masks.append(mask)

N_BLOCKS = len(all_blocks)

# Precompute per-block list of covered subset indices (35 each)
block_bit_indices = []
for mask in all_masks:
    bits = []
    mm = mask
    while mm:
        lsb = mm & -mm
        bits.append(lsb.bit_length() - 1)
        mm ^= lsb
    block_bit_indices.append(bits)

# Weighted greedy weights
weights = [1.0 / f for f in subset_frequencies]
weights_np = np.array(weights, dtype=np.float64)
weights_uniform = np.ones(N_SUBSETS, dtype=np.float64)
weights_sqrt_inv = np.array([1.0 / math.sqrt(f) for f in subset_frequencies], dtype=np.float64)

# Coverage matrix for fast vectorized operations
# coverage_matrix[i, j] = 1 if block i covers subset j
coverage_matrix = np.zeros((N_BLOCKS, N_SUBSETS), dtype=np.int8)
for i, bits in enumerate(block_bit_indices):
    for b in bits:
        coverage_matrix[i, b] = 1

# Precompute for each subset, the list of blocks containing it
subset_to_blocks = [list(blocks_containing_subset[i]) for i in range(N_SUBSETS)]


def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        all_k_subsets = list(itertools.combinations(range(v), k))
        random.shuffle(all_k_subsets)
        for b in all_k_subsets:
            if not uncovered:
                break
            covered = set(itertools.combinations(b, t))
            if uncovered & covered:
                blocks.append(list(b))
                uncovered -= covered
        return blocks

    start_time = time.time()
    deadline = start_time + 112  # Safety margin below 120s kill

    def fast_greedy(weight_vector=None):
        """Vectorized weighted greedy using numpy for speed."""
        if weight_vector is None:
            weight_vector = weights_np
        uncovered = np.ones(N_SUBSETS, dtype=np.int8)
        chosen = []
        remaining = N_SUBSETS

        while remaining > 0:
            gains = coverage_matrix @ (weight_vector * uncovered)
            best_idx = int(np.argmax(gains))
            best_gain = gains[best_idx]

            if best_gain <= 0:
                break

            chosen.append(best_idx)

            newly_covered = coverage_matrix[best_idx]
            still_uncovered = uncovered & (1 - newly_covered)
            remaining = int(np.sum(still_uncovered))
            uncovered = still_uncovered

        return chosen

    # 1. Try multiple greedy initializations with different weightings
    greedy_results = []
    for w_vec in [weights_np, weights_uniform, weights_sqrt_inv]:
        if time.time() > deadline - 10:
            break
        result = fast_greedy(w_vec)
        if result:
            greedy_results.append(result)

    # Pick the best greedy result
    if greedy_results:
        best_greedy = min(greedy_results, key=len)
    else:
        best_greedy = fast_greedy(weights_np)

    current_blocks = best_greedy.copy()
    best_len = len(current_blocks)
    best_blocks = current_blocks.copy()

    # Build counts
    counts = [0] * N_SUBSETS
    for idx in current_blocks:
        for bit in block_bit_indices[idx]:
            counts[bit] += 1

    def is_redundant_pos(pos, cnt):
        idx = current_blocks[pos]
        for bit in block_bit_indices[idx]:
            if cnt[bit] == 1:
                return False
        return True

    def remove_block_pos(pos, cnt):
        idx = current_blocks[pos]
        for bit in block_bit_indices[idx]:
            cnt[bit] -= 1
        current_blocks.pop(pos)

    def add_block_idx(idx, cnt):
        current_blocks.append(idx)
        for bit in block_bit_indices[idx]:
            cnt[bit] += 1

    def prune_all(blocks, cnt):
        """Remove all redundant blocks. Returns count removed."""
        removed = 0
        progress = True
        while progress:
            progress = False
            for pos in range(len(blocks)):
                idx = blocks[pos]
                redundant = True
                for bit in block_bit_indices[idx]:
                    if cnt[bit] == 1:
                        redundant = False
                        break
                if redundant:
                    for bit in block_bit_indices[idx]:
                        cnt[bit] -= 1
                    blocks.pop(pos)
                    removed += 1
                    progress = True
        return removed

    # 2. Local search
    no_improve = 0
    max_no_improve = 50

    while time.time() < deadline:
        # Phase A: prune all redundant blocks
        removed = prune_all(current_blocks, counts)
        if removed > 0:
            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = current_blocks.copy()
            no_improve = 0
            continue

        # Phase B: try 1-for-1 swaps
        swap_done = False
        n_sample = min(len(current_blocks), 40)
        sampled = random.sample(range(len(current_blocks)), n_sample)

        for pos in sampled:
            idx_to_remove = current_blocks[pos]
            # Find unique subsets (covered only by this block)
            unique_bits = [bit for bit in block_bit_indices[idx_to_remove] if counts[bit] == 1]

            if not unique_bits:
                # Redundant, shouldn't happen after prune but handle it
                remove_block_pos(pos, counts)
                swap_done = True
                break

            # Find candidate blocks that cover all unique_bits
            candidates = None
            for bit in unique_bits:
                blocks_with_bit = set(subset_to_blocks[bit])
                if candidates is None:
                    candidates = blocks_with_bit
                else:
                    candidates &= blocks_with_bit

            if candidates:
                candidates.discard(idx_to_remove)
                if candidates:
                    # Pick a random candidate
                    j = random.choice(list(candidates))
                    if j not in current_blocks:
                        # Try the swap
                        temp_counts = counts.copy()
                        for bit in block_bit_indices[idx_to_remove]:
                            temp_counts[bit] -= 1
                        # Check coverage after removal
                        valid = True
                        for bit in block_bit_indices[idx_to_remove]:
                            if temp_counts[bit] == 0 and bit not in block_bit_indices[j]:
                                valid = False
                                break
                        if valid:
                            # Apply swap
                            remove_block_pos(pos, counts)
                            add_block_idx(j, counts)
                            swap_done = True
                            if len(current_blocks) < best_len:
                                best_len = len(current_blocks)
                                best_blocks = current_blocks.copy()
                            break

        if swap_done:
            no_improve = 0
            continue

        # Phase C: try 2-for-1 swaps (remove 2, add 1)
        if len(current_blocks) > 1 and no_improve > 20:
            swap2_done = False
            for _ in range(10):
                i1 = random.randint(0, len(current_blocks) - 1)
                i2 = random.randint(0, len(current_blocks) - 1)
                if i1 == i2:
                    continue

                idx1 = current_blocks[i1]
                idx2 = current_blocks[i2]

                # Find uncovered bits if we remove both
                temp_counts = counts.copy()
                for bit in block_bit_indices[idx1]:
                    temp_counts[bit] -= 1
                for bit in block_bit_indices[idx2]:
                    temp_counts[bit] -= 1

                uncovered_bits = set()
                for bit in set(block_bit_indices[idx1]) | set(block_bit_indices[idx2]):
                    if temp_counts[bit] == 0:
                        uncovered_bits.add(bit)

                if not uncovered_bits:
                    # Both redundant together
                    # Remove the higher index first to keep indices valid
                    if i1 > i2:
                        i1, i2 = i2, i1
                    remove_block_pos(i1, counts)
                    remove_block_pos(i2, counts)
                    swap2_done = True
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = current_blocks.copy()
                    break

                if len(uncovered_bits) <= 6:
                    # Find a single block covering all uncovered bits
                    candidates = None
                    for bit in list(uncovered_bits)[:4]:
                        blocks_with_bit = set(subset_to_blocks[bit])
                        if candidates is None:
                            candidates = blocks_with_bit
                        else:
                            candidates &= blocks_with_bit

                    if candidates:
                        candidates.discard(idx1)
                        candidates.discard(idx2)
                        if candidates:
                            j = random.choice(list(candidates))
                            if j not in current_blocks:
                                # Verify full coverage
                                full_cover = True
                                for bit in uncovered_bits:
                                    if bit not in block_bit_indices[j]:
                                        full_cover = False
                                        break
                                if full_cover:
                                    # Apply
                                    if i1 > i2:
                                        i1, i2 = i2, i1
                                    remove_block_pos(i1, counts)
                                    remove_block_pos(i2, counts)
                                    add_block_idx(j, counts)
                                    swap2_done = True
                                    break

            if swap2_done:
                no_improve = 0
                continue

        # Phase D: perturbation if stuck
        if no_improve > max_no_improve:
            # Add a random block and try to prune
            c = random.randint(0, N_BLOCKS - 1)
            if c not in current_blocks:
                add_block_idx(c, counts)

            # Aggressive prune
            for _ in range(30):
                progress = False
                for pos in range(len(current_blocks)):
                    idx = current_blocks[pos]
                    redundant = True
                    for bit in block_bit_indices[idx]:
                        if counts[bit] == 1:
                            redundant = False
                            break
                    if redundant:
                        remove_block_pos(pos, counts)
                        progress = True
                        if len(current_blocks) < best_len:
                            best_len = len(current_blocks)
                            best_blocks = current_blocks.copy()
                        break
                if not progress:
                    break

            no_improve = 0
        else:
            no_improve += 1

    # Final cleanup
    prune_all(current_blocks, counts)
    if len(current_blocks) < best_len:
        best_len = len(current_blocks)
        best_blocks = current_blocks.copy()

    return [list(all_blocks[i]) for i in best_blocks]
