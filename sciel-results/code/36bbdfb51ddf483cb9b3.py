import itertools
import random
import time
import math
import numpy as np

# Global Precomputation
V = 19
K = 7
T = 4

subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

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

block_bit_indices = []
for mask in all_masks:
    bits = []
    mm = mask
    while mm:
        lsb = mm & -mm
        bits.append(lsb.bit_length() - 1)
        mm ^= lsb
    block_bit_indices.append(bits)

# Coverage matrix
coverage_matrix = np.zeros((N_BLOCKS, N_SUBSETS), dtype=np.int8)
for i, bits in enumerate(block_bit_indices):
    for b in bits:
        coverage_matrix[i, b] = 1

weights_inv_freq = np.array([1.0 / f for f in subset_frequencies], dtype=np.float64)
weights_uniform = np.ones(N_SUBSETS, dtype=np.float64)
weights_sqrt_inv = np.array([1.0 / math.sqrt(f) for f in subset_frequencies], dtype=np.float64)


def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
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
    deadline = start_time + 145  # Safety margin below 150s kill

    def fast_greedy(weight_vector, reweight_iters=3):
        """Weighted greedy with optional dynamic reweighting."""
        uncovered = np.ones(N_SUBSETS, dtype=np.int8)
        chosen = []
        remaining = N_SUBSETS
        w = weight_vector.astype(np.float64).copy()

        while remaining > 0:
            gains = coverage_matrix @ (w * uncovered)
            best_idx = int(np.argmax(gains))
            best_gain = gains[best_idx]
            if best_gain <= 0:
                break
            chosen.append(best_idx)
            newly_covered = coverage_matrix[best_idx]
            still_uncovered = uncovered & (1 - newly_covered)
            remaining = int(np.sum(still_uncovered))
            uncovered = still_uncovered

            # Dynamic reweight: boost subsets that are still uncovered and rare
            if reweight_iters > 0 and remaining > 0 and len(chosen) % 5 == 0:
                # Increase weight for uncovered subsets proportional to 1/sqrt(freq)
                unc_mask = uncovered.astype(np.float64)
                w = w * (1.0 + 0.1 * unc_mask * weights_sqrt_inv)
                # Normalize to prevent overflow
                w = w / (np.sum(w) + 1e-12) * N_SUBSETS

            if time.time() > deadline - 10:
                break
        return chosen

    def local_search(initial_blocks, seed, time_budget=10.0):
        """Local search with 1-for-1 swaps, 2-for-1 swaps, and pruning."""
        rng = random.Random(seed)
        current_blocks = initial_blocks[:]
        counts = [0] * N_SUBSETS
        for idx in current_blocks:
            for bit in block_bit_indices[idx]:
                counts[bit] += 1

        best_len = len(current_blocks)
        best_blocks = current_blocks[:]
        no_improve_count = 0
        max_no_improve = 80
        ls_start = time.time()

        while time.time() < deadline - 5 and (time.time() - ls_start) < time_budget:
            # Phase 1: Prune redundant blocks
            pruned = False
            for i in range(len(current_blocks)):
                if time.time() > deadline - 5:
                    break
                idx = current_blocks[i]
                redundant = True
                for bit in block_bit_indices[idx]:
                    if counts[bit] == 1:
                        redundant = False
                        break
                if redundant:
                    for bit in block_bit_indices[idx]:
                        counts[bit] -= 1
                    current_blocks.pop(i)
                    pruned = True
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = current_blocks[:]
                    no_improve_count = 0
                    break

            if pruned:
                continue

            no_improve_count += 1

            if no_improve_count > max_no_improve:
                # Perturbation: add 1-2 random blocks
                for _ in range(rng.randint(1, 2)):
                    c = rng.randint(0, N_BLOCKS - 1)
                    if c not in current_blocks:
                        current_blocks.append(c)
                        for bit in block_bit_indices[c]:
                            counts[bit] += 1
                no_improve_count = 0
                continue

            # Phase 2: Try 1-for-1 swap
            sample_size = min(30, len(current_blocks))
            best_swap = None
            best_unique = float('inf')

            for _ in range(sample_size):
                idx_check = rng.randint(0, len(current_blocks) - 1)
                idx = current_blocks[idx_check]
                unique_count = 0
                unique_bits = []
                for bit in block_bit_indices[idx]:
                    if counts[bit] == 1:
                        unique_count += 1
                        unique_bits.append(bit)
                if 0 < unique_count < best_unique:
                    best_unique = unique_count
                    best_swap = (idx_check, idx, unique_bits)

            if best_swap is not None:
                idx_check, idx, uncovered_bits = best_swap
                if len(uncovered_bits) <= 6:
                    # Find replacement
                    uncovered_bits.sort(key=lambda b: len(blocks_containing_subset[b]))
                    candidates = list(blocks_containing_subset[uncovered_bits[0]])
                    for b in uncovered_bits[1:]:
                        if not candidates:
                            break
                        set_cands = set(candidates)
                        set_b = set(blocks_containing_subset[b])
                        candidates = list(set_cands.intersection(set_b))

                    valid_candidates = [c for c in candidates if c != idx]
                    if valid_candidates:
                        # Pick candidate that minimizes new unique subsets created
                        best_j = valid_candidates[0]
                        best_new_unique = float('inf')
                        search_limit = min(40, len(valid_candidates))
                        for j in valid_candidates[:search_limit]:
                            # Count how many subsets j covers that would become unique
                            new_unique = 0
                            for bit in block_bit_indices[j]:
                                if counts[bit] == 1 and bit not in uncovered_bits:
                                    new_unique += 1
                            if new_unique < best_new_unique:
                                best_new_unique = new_unique
                                best_j = j

                        # Perform swap
                        for bit in block_bit_indices[idx]:
                            counts[bit] -= 1
                        current_blocks.pop(idx_check)
                        current_blocks.append(best_j)
                        for bit in block_bit_indices[best_j]:
                            counts[bit] += 1
                        no_improve_count = 0
                        continue

            # Phase 3: Try 2-for-1 swap (remove 2 blocks, add 1)
            if len(current_blocks) >= 3:
                # Sample pairs of blocks
                for _ in range(min(10, len(current_blocks) // 2)):
                    i1 = rng.randint(0, len(current_blocks) - 1)
                    i2 = rng.randint(0, len(current_blocks) - 1)
                    if i1 == i2:
                        continue
                    idx1 = current_blocks[i1]
                    idx2 = current_blocks[i2]

                    # Find subsets that would become uncovered
                    # A subset is uniquely covered by idx1 or idx2 if count==1 and it's in that block
                    unique_to_1 = [b for b in block_bit_indices[idx1] if counts[b] == 1]
                    unique_to_2 = [b for b in block_bit_indices[idx2] if counts[b] == 1]
                    # Also subsets covered by both where count==2 would become count==1 (not uncovered)
                    # Only truly uncovered: count==1 subsets

                    all_unique = set(unique_to_1) | set(unique_to_2)
                    if len(all_unique) <= 8:
                        # Find a block that covers all these unique subsets
                        all_unique_list = sorted(all_unique, key=lambda b: len(blocks_containing_subset[b]))
                        if all_unique_list:
                            candidates = list(blocks_containing_subset[all_unique_list[0]])
                            for b in all_unique_list[1:]:
                                if not candidates:
                                    break
                                set_cands = set(candidates)
                                set_b = set(blocks_containing_subset[b])
                                candidates = list(set_cands.intersection(set_b))

                            valid_candidates = [c for c in candidates if c != idx1 and c != idx2]
                            if valid_candidates:
                                # Pick one that minimizes new unique subsets
                                best_j = valid_candidates[0]
                                best_new_unique = float('inf')
                                for j in valid_candidates[:20]:
                                    new_unique = 0
                                    for bit in block_bit_indices[j]:
                                        if counts[bit] == 1 and bit not in all_unique:
                                            new_unique += 1
                                    if new_unique < best_new_unique:
                                        best_new_unique = new_unique
                                        best_j = j

                                # Perform 2-for-1 swap
                                for bit in block_bit_indices[idx1]:
                                    counts[bit] -= 1
                                for bit in block_bit_indices[idx2]:
                                    counts[bit] -= 1
                                # Remove both (handle index order)
                                if i1 < i2:
                                    current_blocks.pop(i2)
                                    current_blocks.pop(i1)
                                else:
                                    current_blocks.pop(i1)
                                    current_blocks.pop(i2)
                                current_blocks.append(best_j)
                                for bit in block_bit_indices[best_j]:
                                    counts[bit] += 1
                                no_improve_count = 0
                                break

        return best_blocks, best_len

    # Run multiple greedy initializations with different strategies
    best_overall_blocks = None
    best_overall_len = float('inf')

    # Strategy 1: Inverse frequency greedy with reweighting
    if time.time() < deadline - 40:
        init_blocks = fast_greedy(weights_inv_freq, reweight_iters=3)
        if init_blocks:
            ls_blocks, ls_len = local_search(init_blocks, seed=42, time_budget=25.0)
            if ls_len < best_overall_len:
                best_overall_len = ls_len
                best_overall_blocks = ls_blocks[:]

    # Strategy 2: Uniform greedy with reweighting
    if time.time() < deadline - 30:
        init_blocks = fast_greedy(weights_uniform, reweight_iters=3)
        if init_blocks:
            ls_blocks, ls_len = local_search(init_blocks, seed=123, time_budget=20.0)
            if ls_len < best_overall_len:
                best_overall_len = ls_len
                best_overall_blocks = ls_blocks[:]

    # Strategy 3: Sqrt inverse frequency
    if time.time() < deadline - 25:
        init_blocks = fast_greedy(weights_sqrt_inv, reweight_iters=3)
        if init_blocks:
            ls_blocks, ls_len = local_search(init_blocks, seed=777, time_budget=20.0)
            if ls_len < best_overall_len:
                best_overall_len = ls_len
                best_overall_blocks = ls_blocks[:]

    # Strategy 4: Randomized greedy (random tie-breaking)
    if time.time() < deadline - 20:
        rng_init = random.Random(999)
        uncovered = np.ones(N_SUBSETS, dtype=np.int8)
        chosen = []
        remaining = N_SUBSETS
        while remaining > 0 and time.time() < deadline - 15:
            gains = coverage_matrix @ (weights_inv_freq * uncovered)
            # Among top candidates, pick randomly
            top_n = min(5, N_BLOCKS)
            top_indices = np.argpartition(gains, -top_n)[-top_n:]
            top_gains = gains[top_indices]
            max_gain = np.max(top_gains)
            # Pick from those with gain close to max
            candidates = top_indices[top_gains >= max_gain * 0.9]
            best_idx = int(rng_init.choice(candidates))
            if gains[best_idx] <= 0:
                break
            chosen.append(best_idx)
            newly_covered = coverage_matrix[best_idx]
            still_uncovered = uncovered & (1 - newly_covered)
            remaining = int(np.sum(still_uncovered))
            uncovered = still_uncovered
        if chosen:
            ls_blocks, ls_len = local_search(chosen, seed=555, time_budget=20.0)
            if ls_len < best_overall_len:
                best_overall_len = ls_len
                best_overall_blocks = ls_blocks[:]

    # Strategy 5: Another random restart
    if time.time() < deadline - 15:
        rng_init = random.Random(2024)
        uncovered = np.ones(N_SUBSETS, dtype=np.int8)
        chosen = []
        remaining = N_SUBSETS
        while remaining > 0 and time.time() < deadline - 10:
            gains = coverage_matrix @ (weights_sqrt_inv * uncovered)
            top_n = min(5, N_BLOCKS)
            top_indices = np.argpartition(gains, -top_n)[-top_n:]
            top_gains = gains[top_indices]
            max_gain = np.max(top_gains)
            candidates = top_indices[top_gains >= max_gain * 0.85]
            best_idx = int(rng_init.choice(candidates))
            if gains[best_idx] <= 0:
                break
            chosen.append(best_idx)
            newly_covered = coverage_matrix[best_idx]
            still_uncovered = uncovered & (1 - newly_covered)
            remaining = int(np.sum(still_uncovered))
            uncovered = still_uncovered
        if chosen:
            ls_blocks, ls_len = local_search(chosen, seed=3141, time_budget=15.0)
            if ls_len < best_overall_len:
                best_overall_len = ls_len
                best_overall_blocks = ls_blocks[:]

    if best_overall_blocks is None:
        init_blocks = fast_greedy(weights_inv_freq)
        best_overall_blocks = init_blocks

    # Final cleanup: remove any remaining redundant blocks
    final_counts = [0] * N_SUBSETS
    for idx in best_overall_blocks:
        for bit in block_bit_indices[idx]:
            final_counts[bit] += 1

    final_blocks = best_overall_blocks[:]
    for i in range(len(final_blocks) - 1, -1, -1):
        idx = final_blocks[i]
        redundant = True
        for bit in block_bit_indices[idx]:
            if final_counts[bit] == 1:
                redundant = False
                break
        if redundant:
            for bit in block_bit_indices[idx]:
                final_counts[bit] -= 1
            final_blocks.pop(i)

    # One more pass of local search on the final result if time permits
    if time.time() < deadline - 5 and len(final_blocks) > 0:
        ls_blocks, ls_len = local_search(final_blocks, seed=9999, time_budget=max(0, deadline - 5 - time.time() - 1))
        if ls_len < len(final_blocks):
            final_blocks = ls_blocks

    return [list(all_blocks[i]) for i in final_blocks]
