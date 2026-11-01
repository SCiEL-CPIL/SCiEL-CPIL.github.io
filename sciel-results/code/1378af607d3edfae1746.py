import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # Precompute t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(t_subsets)}

    # Precompute k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)

    # Build bitmasks for each block and reverse mapping
    block_bitmasks = [0] * n_k_blocks
    t_to_k = [[] for _ in range(n_t_subsets)]
    for i, block in enumerate(all_k_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            mask |= (1 << idx)
            t_to_k[idx].append(i)
        block_bitmasks[i] = mask

    full_mask = (1 << n_t_subsets) - 1

    # Precompute popcount of each block's mask (initial gain)
    block_popcounts = [m.bit_count() for m in block_bitmasks]

    def greedy_cover(randomize=False, seed=None):
        """Fast greedy set cover with incremental gain updates."""
        if seed is not None:
            random.seed(seed)

        uncovered = full_mask
        selected_blocks = []
        used_set = set()

        # gain[i] = number of uncovered t-subsets covered by block i
        gain = list(block_popcounts)

        while uncovered:
            if time.time() > deadline:
                break

            # Find max gain among unused blocks
            best_idx = -1
            best_gain = 0
            candidates = []

            for i in range(n_k_blocks):
                if i in used_set:
                    continue
                g = gain[i]
                if g > best_gain:
                    best_gain = g
                    candidates = [i]
                elif g == best_gain and g > 0:
                    candidates.append(i)

            if best_gain == 0:
                break

            if randomize and candidates:
                best_idx = random.choice(candidates)
            else:
                best_idx = candidates[0]

            selected_blocks.append(best_idx)
            used_set.add(best_idx)

            # Update: newly covered t-subsets
            newly_covered = uncovered & block_bitmasks[best_idx]
            uncovered &= ~block_bitmasks[best_idx]

            # Decrement gain for all blocks that covered any newly-covered t-subset
            nc = newly_covered
            while nc:
                lsb = nc & -nc
                j = lsb.bit_length() - 1
                nc ^= lsb
                for blk in t_to_k[j]:
                    if blk not in used_set:
                        gain[blk] -= 1

        coverage_counts = np.zeros(n_t_subsets, dtype=np.int32)
        for b_idx in selected_blocks:
            mask = block_bitmasks[b_idx]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                coverage_counts[idx] += 1
                m ^= lsb

        return selected_blocks, coverage_counts

    def local_search_remove(blocks_indices, coverage_counts):
        """Remove redundant blocks iteratively."""
        if time.time() > deadline:
            return blocks_indices, coverage_counts

        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            to_remove = []

            for b_idx in blocks_indices:
                mask = block_bitmasks[b_idx]
                m = mask
                essential = False
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if coverage_counts[idx] == 1:
                        essential = True
                        break
                    m ^= lsb
                if not essential:
                    to_remove.append(b_idx)

            if to_remove:
                remove_set = set(to_remove)
                blocks_indices = [b for b in blocks_indices if b not in remove_set]
                for b_idx in to_remove:
                    mask = block_bitmasks[b_idx]
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        coverage_counts[idx] -= 1
                        m ^= lsb
                changed = True

        return blocks_indices, coverage_counts

    def local_search_swap(blocks_indices, coverage_counts):
        """Try to swap blocks to reduce count."""
        if time.time() > deadline:
            return blocks_indices, coverage_counts

        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False

            for i, b_idx in enumerate(blocks_indices):
                if time.time() > deadline:
                    break

                mask_to_remove = block_bitmasks[b_idx]
                m = mask_to_remove
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    coverage_counts[idx] -= 1
                    m ^= lsb

                uncovered_mask = 0
                for j in range(n_t_subsets):
                    if coverage_counts[j] == 0:
                        uncovered_mask |= (1 << j)

                if uncovered_mask == 0:
                    blocks_indices = [b for j, b in enumerate(blocks_indices) if j != i]
                    improved = True
                    break

                best_new_idx = -1
                best_new_gain = 0
                for new_idx in range(n_k_blocks):
                    if new_idx == b_idx or new_idx in blocks_indices:
                        continue
                    gain_val = (uncovered_mask & block_bitmasks[new_idx]).bit_count()
                    if gain_val > best_new_gain:
                        best_new_gain = gain_val
                        best_new_idx = new_idx

                m = mask_to_remove
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    coverage_counts[idx] += 1
                    m ^= lsb

                if best_new_idx != -1 and best_new_gain >= uncovered_mask.bit_count():
                    m = mask_to_remove
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        coverage_counts[idx] -= 1
                        m ^= lsb

                    blocks_indices[i] = best_new_idx

                    mask_to_add = block_bitmasks[best_new_idx]
                    m = mask_to_add
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        coverage_counts[idx] += 1
                        m ^= lsb

                    improved = True
                    break

        return blocks_indices, coverage_counts

    def compute_coverage_mask(blocks_list):
        mask = 0
        for b_idx in blocks_list:
            mask |= block_bitmasks[b_idx]
        return mask

    def cover_holes_greedy(existing_blocks, holes_mask):
        new_blocks = []
        uncovered = holes_mask
        used = set(existing_blocks)
        gain = [0] * n_k_blocks
        for i in range(n_k_blocks):
            gain[i] = (uncovered & block_bitmasks[i]).bit_count()

        while uncovered:
            if time.time() > deadline:
                break
            best_idx = -1
            best_gain = 0
            for i in range(n_k_blocks):
                if i in used:
                    continue
                if gain[i] > best_gain:
                    best_gain = gain[i]
                    best_idx = i
            if best_idx == -1:
                break
            new_blocks.append(best_idx)
            used.add(best_idx)
            newly = uncovered & block_bitmasks[best_idx]
            uncovered &= ~block_bitmasks[best_idx]
            nc = newly
            while nc:
                lsb = nc & -nc
                j = lsb.bit_length() - 1
                nc ^= lsb
                for blk in t_to_k[j]:
                    if blk not in used:
                        gain[blk] -= 1
        return new_blocks

    def rebuild_counts(blocks_list):
        counts = np.zeros(n_t_subsets, dtype=np.int32)
        for b_idx in blocks_list:
            mask = block_bitmasks[b_idx]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
        return counts

    # === Phase 1: Multiple diverse greedy starts ===
    best_blocks = None
    best_counts = None
    best_count = float('inf')

    # Deterministic greedy
    blocks, counts = greedy_cover(randomize=False)
    blocks, counts = local_search_remove(blocks, counts)
    blocks, counts = local_search_swap(blocks, counts)
    best_blocks = blocks
    best_counts = counts
    best_count = len(blocks)

    # Randomized greedy restarts (fast due to incremental gain)
    n_restarts = 8
    for r in range(n_restarts):
        if time.time() > deadline - 20:
            break
        seed = 42 + r * 137
        blocks, counts = greedy_cover(randomize=True, seed=seed)
        blocks, counts = local_search_remove(blocks, counts)
        blocks, counts = local_search_swap(blocks, counts)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_counts = counts
            best_count = len(blocks)

    # === Phase 2: Perturbation-based local search ===
    attempts = 0
    while time.time() < deadline - 10:
        attempts += 1

        num_remove = min(len(best_blocks), random.randint(3, 8))
        if num_remove == 0:
            break

        to_remove = random.sample(best_blocks, num_remove)
        remove_set = set(to_remove)
        perturbed_blocks = [b for b in best_blocks if b not in remove_set]

        covered_mask = compute_coverage_mask(perturbed_blocks)
        holes_mask = full_mask & ~covered_mask

        added_blocks = cover_holes_greedy(perturbed_blocks, holes_mask)
        new_total_blocks = perturbed_blocks + added_blocks
        new_count = len(new_total_blocks)

        if new_count < best_count:
            new_counts = rebuild_counts(new_total_blocks)
            cleaned_blocks, cleaned_counts = local_search_remove(new_total_blocks, new_counts)
            cleaned_blocks, cleaned_counts = local_search_swap(cleaned_blocks, cleaned_counts)

            if len(cleaned_blocks) < best_count:
                best_blocks = cleaned_blocks
                best_counts = cleaned_counts
                best_count = len(best_blocks)
            elif new_count < best_count:
                best_blocks = new_total_blocks
                best_counts = new_counts
                best_count = len(best_blocks)

    # Final local search pass
    if time.time() < deadline - 5:
        best_blocks, best_counts = local_search_remove(best_blocks, best_counts)
        best_blocks, best_counts = local_search_swap(best_blocks, best_counts)

    # Convert to output format
    final_blocks = [list(all_k_blocks[i]) for i in best_blocks]
    return final_blocks
