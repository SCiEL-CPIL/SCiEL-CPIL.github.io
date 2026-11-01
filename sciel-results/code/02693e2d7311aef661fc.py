import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit

    # Precompute t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(t_subsets)}

    # Precompute k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)

    # Build bitmasks for each block
    block_bitmasks = [0] * n_k_blocks
    for i, block in enumerate(all_k_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            mask |= (1 << idx)
        block_bitmasks[i] = mask

    # Precompute list of covered t-subset indices for each block (for fast iteration)
    block_covered_indices = [None] * n_k_blocks
    for i in range(n_k_blocks):
        mask = block_bitmasks[i]
        indices = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            indices.append(idx)
            m ^= lsb
        block_covered_indices[i] = indices

    full_mask = (1 << n_t_subsets) - 1

    def greedy_cover_weighted():
        """Weighted greedy set cover with dual variable updates."""
        uncovered_mask = full_mask
        selected_blocks = []
        coverage_counts = np.zeros(n_t_subsets, dtype=np.int32)
        # Dual variables: penalty for selecting blocks that overlap heavily
        dual = np.zeros(n_k_blocks, dtype=np.float64)
        alpha = 0.1  # Dual update rate

        while uncovered_mask:
            if time.time() > deadline:
                break

            best_idx = -1
            best_score = -1.0

            # Find block with max (marginal_gain - dual_penalty)
            for i in range(n_k_blocks):
                gain = (uncovered_mask & block_bitmasks[i]).bit_count()
                if gain == 0:
                    continue
                score = gain - alpha * dual[i]
                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx == -1:
                break

            selected_blocks.append(best_idx)

            # Update coverage counts
            for idx in block_covered_indices[best_idx]:
                coverage_counts[idx] += 1

            uncovered_mask &= ~block_bitmasks[best_idx]

            # Update dual variables: increase penalty for blocks that overlap with selected
            if alpha > 0:
                for j in range(n_k_blocks):
                    overlap = (block_bitmasks[j] & block_bitmasks[best_idx]).bit_count()
                    if overlap > 0:
                        dual[j] += alpha * overlap / n_t_subsets

        return selected_blocks, coverage_counts

    def greedy_cover_standard():
        """Standard greedy set cover."""
        uncovered_mask = full_mask
        selected_blocks = []
        coverage_counts = np.zeros(n_t_subsets, dtype=np.int32)

        while uncovered_mask:
            if time.time() > deadline:
                break

            best_idx = -1
            best_gain = 0

            for i in range(n_k_blocks):
                gain = (uncovered_mask & block_bitmasks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i

            if best_idx == -1:
                break

            selected_blocks.append(best_idx)
            for idx in block_covered_indices[best_idx]:
                coverage_counts[idx] += 1
            uncovered_mask &= ~block_bitmasks[best_idx]

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
                essential = False
                for idx in block_covered_indices[b_idx]:
                    if coverage_counts[idx] == 1:
                        essential = True
                        break
                if not essential:
                    to_remove.append(b_idx)

            if to_remove:
                remove_set = set(to_remove)
                blocks_indices = [b for b in blocks_indices if b not in remove_set]
                for b_idx in to_remove:
                    for idx in block_covered_indices[b_idx]:
                        coverage_counts[idx] -= 1
                changed = True

        return blocks_indices, coverage_counts

    def local_search_swap(blocks_indices, coverage_counts):
        """Try to swap blocks: remove one, add one that covers more uncovered."""
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

                # Temporarily remove b_idx
                for idx in block_covered_indices[b_idx]:
                    coverage_counts[idx] -= 1

                # Find uncovered t-subsets
                uncovered_list = [j for j in range(n_t_subsets) if coverage_counts[j] == 0]
                n_uncovered = len(uncovered_list)

                if n_uncovered == 0:
                    # Can remove b_idx entirely
                    blocks_indices = [b for j, b in enumerate(blocks_indices) if j != i]
                    improved = True
                    break

                # Try to find a block that covers the most uncovered
                best_new_idx = -1
                best_new_gain = 0
                for new_idx in range(n_k_blocks):
                    if new_idx == b_idx or new_idx in blocks_indices:
                        continue
                    gain = sum(1 for j in uncovered_list if (block_bitmasks[new_idx] >> j) & 1)
                    if gain > best_new_gain:
                        best_new_gain = gain
                        best_new_idx = new_idx

                # Restore b_idx
                for idx in block_covered_indices[b_idx]:
                    coverage_counts[idx] += 1

                # If we can cover all uncovered with one new block, do the swap
                if best_new_idx != -1 and best_new_gain >= n_uncovered:
                    # Remove old block
                    for idx in block_covered_indices[b_idx]:
                        coverage_counts[idx] -= 1

                    blocks_indices[i] = best_new_idx

                    # Add new block
                    for idx in block_covered_indices[best_new_idx]:
                        coverage_counts[idx] += 1

                    improved = True
                    break

        return blocks_indices, coverage_counts

    def local_search_2swap(blocks_indices, coverage_counts):
        """Try 2-opt: remove two blocks, add two blocks that cover better."""
        if time.time() > deadline:
            return blocks_indices, coverage_counts

        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False

            n = len(blocks_indices)
            if n < 2:
                break

            # Sample a few pairs to try (not all pairs for speed)
            num_pairs_to_try = min(50, n * (n - 1) // 2)
            pairs = random.sample(range(n * (n - 1) // 2), min(num_pairs_to_try, n * (n - 1) // 2))

            for pair_idx in pairs:
                if time.time() > deadline:
                    break

                # Convert pair_idx to (i, j)
                count = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        if count == pair_idx:
                            break
                        count += 1
                    if count == pair_idx:
                        break

                b1, b2 = blocks_indices[i], blocks_indices[j]

                # Temporarily remove both
                for idx in block_covered_indices[b1]:
                    coverage_counts[idx] -= 1
                for idx in block_covered_indices[b2]:
                    coverage_counts[idx] -= 1

                # Find uncovered
                uncovered_list = [j for j in range(n_t_subsets) if coverage_counts[j] == 0]
                n_uncovered = len(uncovered_list)

                if n_uncovered == 0:
                    # Can remove both
                    blocks_indices = [b for idx, b in enumerate(blocks_indices) if idx not in (i, j)]
                    improved = True
                    break

                # Try to find two blocks that cover all uncovered
                best_pair = None
                best_total_gain = 0

                # Sample candidate new blocks
                candidates = random.sample(range(n_k_blocks), min(200, n_k_blocks))
                for c1 in candidates:
                    if c1 == b1 or c1 == b2 or c1 in blocks_indices:
                        continue
                    gain1 = sum(1 for j in uncovered_list if (block_bitmasks[c1] >> j) & 1)
                    if gain1 < n_uncovered // 2:
                        continue
                    for c2 in candidates:
                        if c2 == b1 or c2 == b2 or c2 in blocks_indices or c2 == c1:
                            continue
                        # Combined coverage
                        mask_c1 = block_bitmasks[c1]
                        mask_c2 = block_bitmasks[c2]
                        combined = mask_c1 | mask_c2
                        gain = sum(1 for j in uncovered_list if (combined >> j) & 1)
                        if gain > best_total_gain:
                            best_total_gain = gain
                            best_pair = (c1, c2)
                        if gain >= n_uncovered:
                            break
                    if best_total_gain >= n_uncovered:
                        break

                # Restore
                for idx in block_covered_indices[b1]:
                    coverage_counts[idx] += 1
                for idx in block_covered_indices[b2]:
                    coverage_counts[idx] += 1

                if best_pair and best_total_gain >= n_uncovered:
                    c1, c2 = best_pair
                    # Remove old blocks
                    for idx in block_covered_indices[b1]:
                        coverage_counts[idx] -= 1
                    for idx in block_covered_indices[b2]:
                        coverage_counts[idx] -= 1
                    # Add new blocks
                    for idx in block_covered_indices[c1]:
                        coverage_counts[idx] += 1
                    for idx in block_covered_indices[c2]:
                        coverage_counts[idx] += 1
                    blocks_indices[i] = c1
                    blocks_indices[j] = c2
                    improved = True
                    break

        return blocks_indices, coverage_counts

    def compute_coverage_mask(blocks_list):
        """Compute the coverage mask for a list of block indices."""
        mask = 0
        for b_idx in blocks_list:
            mask |= block_bitmasks[b_idx]
        return mask

    def cover_holes_greedy(existing_blocks, holes_mask):
        """Cover holes using greedy, returning new blocks added."""
        new_blocks = []
        uncovered = holes_mask
        existing_set = set(existing_blocks)
        while uncovered:
            if time.time() > deadline:
                break
            best_idx = -1
            best_gain = 0
            for i in range(n_k_blocks):
                if i in existing_set:
                    continue
                gain = (uncovered & block_bitmasks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1:
                break
            new_blocks.append(best_idx)
            existing_set.add(best_idx)
            uncovered &= ~block_bitmasks[best_idx]
        return new_blocks

    def rebuild_counts(blocks_list):
        """Rebuild coverage counts from scratch."""
        counts = np.zeros(n_t_subsets, dtype=np.int32)
        for b_idx in blocks_list:
            for idx in block_covered_indices[b_idx]:
                counts[idx] += 1
        return counts

    def full_local_search(blocks, counts):
        """Run all local search passes."""
        blocks, counts = local_search_remove(blocks, counts)
        blocks, counts = local_search_swap(blocks, counts)
        blocks, counts = local_search_remove(blocks, counts)
        blocks, counts = local_search_swap(blocks, counts)
        if time.time() < deadline - 10:
            blocks, counts = local_search_2swap(blocks, counts)
        blocks, counts = local_search_remove(blocks, counts)
        return blocks, counts

    # Try both standard and weighted greedy, keep the better
    best_blocks = None
    best_counts = None
    best_count = float('inf')

    # Standard greedy
    blocks1, counts1 = greedy_cover_standard()
    blocks1, counts1 = full_local_search(blocks1, counts1)
    if len(blocks1) < best_count:
        best_blocks = blocks1
        best_counts = counts1
        best_count = len(blocks1)

    # Weighted greedy
    if time.time() < deadline - 20:
        blocks2, counts2 = greedy_cover_weighted()
        blocks2, counts2 = full_local_search(blocks2, counts2)
        if len(blocks2) < best_count:
            best_blocks = blocks2
            best_counts = counts2
            best_count = len(blocks2)

    # Perturbation-based local search
    attempts = 0
    while time.time() < deadline - 15:
        attempts += 1

        # Targeted perturbation: remove blocks with lowest "uniqueness"
        # Compute how many t-subsets each block uniquely covers
        num_remove = min(len(best_blocks), random.randint(3, 8))
        if num_remove == 0:
            break

        # Score blocks by how many t-subsets they uniquely cover
        block_uniqueness = []
        for b_idx in best_blocks:
            unique_count = 0
            for idx in block_covered_indices[b_idx]:
                if best_counts[idx] == 1:
                    unique_count += 1
            block_uniqueness.append((unique_count, b_idx))

        # Remove from lowest uniqueness first (with some randomness)
        random.shuffle(block_uniqueness)
        to_remove = [b for _, b in block_uniqueness[:num_remove]]
        remove_set = set(to_remove)
        perturbed_blocks = [b for b in best_blocks if b not in remove_set]

        # Compute holes
        covered_mask = compute_coverage_mask(perturbed_blocks)
        holes_mask = full_mask & ~covered_mask

        # Cover holes
        added_blocks = cover_holes_greedy(perturbed_blocks, holes_mask)

        new_total_blocks = perturbed_blocks + added_blocks
        new_count = len(new_total_blocks)

        if new_count < best_count:
            # Rebuild counts and run local search
            new_counts = rebuild_counts(new_total_blocks)
            cleaned_blocks, cleaned_counts = full_local_search(new_total_blocks, new_counts)

            if len(cleaned_blocks) < best_count:
                best_blocks = cleaned_blocks
                best_counts = cleaned_counts
                best_count = len(best_blocks)
            elif len(new_total_blocks) < best_count:
                best_blocks = new_total_blocks
                best_counts = new_counts
                best_count = len(best_blocks)

    # Final local search pass
    if time.time() < deadline - 5:
        best_blocks, best_counts = full_local_search(best_blocks, best_counts)

    # Convert to output format
    final_blocks = [list(all_k_blocks[i]) for i in best_blocks]

    return final_blocks
