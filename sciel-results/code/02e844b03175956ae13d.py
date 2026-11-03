import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {fs: i for i, fs in enumerate(all_tsubsets)}
    n_tsubsets = len(all_tsubsets)

    def get_block_tindices(block):
        """Returns list of indices of t-subsets covered by block."""
        indices = []
        for c in itertools.combinations(block, t):
            indices.append(t_subset_to_idx[frozenset(c)])
        return indices

    def greedy_cover():
        counts = np.zeros(n_tsubsets, dtype=np.int32)
        uncovered = n_tsubsets
        blocks = []

        while uncovered > 0 and time.time() - start_time < time_limit - 5:
            unc_indices = np.where(counts == 0)[0]
            if len(unc_indices) == 0:
                break

            seed_idx = unc_indices[random.randint(0, len(unc_indices) - 1)]
            seed_triple = all_tsubsets[seed_idx]

            remaining_points = [x for x in range(v) if x not in seed_triple]
            candidates = []
            for comb in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(list(seed_triple) + list(comb)))
                candidates.append(block)

            best_gain = -1
            best_block = None
            best_block_t_indices = None

            for b in candidates:
                t_indices = get_block_tindices(b)
                gain = 0
                for idx in t_indices:
                    if counts[idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
                    best_block_t_indices = t_indices

            if best_block is None:
                break

            blocks.append(best_block)
            for idx in best_block_t_indices:
                if counts[idx] == 0:
                    uncovered -= 1
                counts[idx] += 1

        return blocks, counts, uncovered

    def local_search(blocks, counts, uncovered):
        best_blocks = list(blocks)
        best_count = len(blocks)

        block_t_indices = [get_block_tindices(b) for b in blocks]

        no_improve = 0
        max_no_improve = 100

        while time.time() - start_time < time_limit - 2:
            improved = False

            # 1. Deletion Pass
            i = 0
            while i < len(blocks):
                t_indices = block_t_indices[i]
                can_remove = True
                for idx in t_indices:
                    if counts[idx] <= 1:
                        can_remove = False
                        break

                if can_remove:
                    for idx in t_indices:
                        counts[idx] -= 1
                    blocks.pop(i)
                    block_t_indices.pop(i)
                    improved = True
                    best_count = len(blocks)
                    best_blocks = list(blocks)
                else:
                    i += 1

            if improved:
                no_improve = 0
                continue

            # 2. Swap Pass
            if len(blocks) == 0:
                break

            i = random.randint(0, len(blocks) - 1)
            old_block = blocks[i]
            old_t_indices = block_t_indices[i]

            critical_indices = [idx for idx in old_t_indices if counts[idx] == 1]

            if not critical_indices:
                no_improve += 1
                continue

            seed_idx = critical_indices[0]
            seed_triple = all_tsubsets[seed_idx]

            remaining_points = [x for x in range(v) if x not in seed_triple]
            candidates = []
            for comb in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(list(seed_triple) + list(comb)))
                candidates.append(block)

            best_new_block = None
            best_new_t_indices = None
            best_score = -1000

            old_t_set = set(old_t_indices)

            for b in candidates:
                if b == old_block:
                    continue
                b_t_indices = get_block_tindices(b)
                b_t_set = set(b_t_indices)

                valid = True
                for idx in critical_indices:
                    if idx not in b_t_set:
                        valid = False
                        break
                if not valid:
                    continue

                score = 0
                for idx in b_t_indices:
                    if idx not in old_t_set:
                        if counts[idx] > 0:
                            score += 1
                        else:
                            score -= 10

                if score > best_score:
                    best_score = score
                    best_new_block = b
                    best_new_t_indices = b_t_indices

            if best_new_block is not None:
                # Apply swap
                for idx in old_t_indices:
                    counts[idx] -= 1
                    if counts[idx] == 0:
                        uncovered += 1
                for idx in best_new_t_indices:
                    if counts[idx] == 0:
                        uncovered -= 1
                    counts[idx] += 1

                blocks[i] = best_new_block
                block_t_indices[i] = best_new_t_indices
                improved = True
                no_improve = 0

            if not improved:
                no_improve += 1
                if no_improve > max_no_improve:
                    # Perturbation
                    if len(blocks) > 1 and time.time() - start_time < time_limit - 10:
                        old_blocks = list(blocks)
                        old_counts = counts.copy()
                        old_uncovered = uncovered
                        old_bt = list(block_t_indices)

                        idx_to_remove = random.randint(0, len(blocks) - 1)
                        rb = blocks[idx_to_remove]
                        rb_t = block_t_indices[idx_to_remove]
                        for idx in rb_t:
                            counts[idx] -= 1
                            if counts[idx] == 0:
                                uncovered += 1
                        blocks.pop(idx_to_remove)
                        block_t_indices.pop(idx_to_remove)

                        # Repair
                        while uncovered > 0 and time.time() - start_time < time_limit - 5:
                            unc_indices = np.where(counts == 0)[0]
                            if len(unc_indices) == 0:
                                break
                            seed_idx = unc_indices[random.randint(0, len(unc_indices) - 1)]
                            seed_triple = all_tsubsets[seed_idx]

                            remaining_points = [x for x in range(v) if x not in seed_triple]
                            candidates = []
                            for comb in itertools.combinations(remaining_points, k - t):
                                block = tuple(sorted(list(seed_triple) + list(comb)))
                                candidates.append(block)

                            best_gain = -1
                            best_b = None
                            best_b_t = None
                            for b in candidates:
                                t_inds = get_block_tindices(b)
                                gain = 0
                                for idx in t_inds:
                                    if counts[idx] == 0:
                                        gain += 1
                                if gain > best_gain:
                                    best_gain = gain
                                    best_b = b
                                    best_b_t = t_inds

                            if best_b is None:
                                break

                            blocks.append(best_b)
                            block_t_indices.append(best_b_t)
                            for idx in best_b_t:
                                if counts[idx] == 0:
                                    uncovered -= 1
                                counts[idx] += 1

                        if uncovered == 0 and len(blocks) < best_count:
                            best_count = len(blocks)
                            best_blocks = list(blocks)
                            no_improve = 0
                        else:
                            blocks = old_blocks
                            counts = old_counts
                            uncovered = old_uncovered
                            block_t_indices = old_bt
                            no_improve += 1
                    else:
                        no_improve += 1

        return best_blocks

    best_overall = None
    best_overall_count = float('inf')

    # Multiple restarts
    for _ in range(3):
        if time.time() - start_time > time_limit - 20:
            break

        blocks, counts, uncovered = greedy_cover()

        if uncovered == 0:
            improved_blocks = local_search(blocks, counts, uncovered)
            if len(improved_blocks) < best_overall_count:
                best_overall_count = len(improved_blocks)
                best_overall = improved_blocks

    if best_overall is None:
        blocks, counts, uncovered = greedy_cover()
        best_overall = blocks

    return [list(b) for b in best_overall]
