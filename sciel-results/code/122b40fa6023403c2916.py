import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 138.0  # safety margin before 150s kill

    points = range(v)
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}

    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)

    # Precompute block_to_subsets and subset_to_blocks
    block_to_subsets = [[] for _ in range(num_blocks)]
    subset_to_blocks = [[] for _ in range(num_subsets)]

    for b_idx, block in enumerate(all_blocks):
        for s in itertools.combinations(block, t):
            s_idx = subset_to_idx[s]
            block_to_subsets[b_idx].append(s_idx)
            subset_to_blocks[s_idx].append(b_idx)

    # Precompute numpy arrays for fast operations
    block_subset_arrays = [np.array(subs, dtype=np.int32) for subs in block_to_subsets]
    subset_block_arrays = [np.array(blocks, dtype=np.int32) for blocks in subset_to_blocks]

    def run_greedy_full(random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()

        uncovered = np.ones(num_subsets, dtype=bool)
        gains = np.array([len(subs) for subs in block_to_subsets], dtype=np.int32)

        selected_blocks = []

        while True:
            if time.time() > deadline - 25.0:
                break
            if not np.any(uncovered):
                break

            noise = rng.uniform(0, 0.3, size=num_blocks)
            scores = gains + noise
            best_idx = int(np.argmax(scores))

            if gains[best_idx] == 0:
                break

            selected_blocks.append(best_idx)

            b_subs = block_subset_arrays[best_idx]
            newly_covered = b_subs[uncovered[b_subs]]

            if len(newly_covered) == 0:
                continue

            uncovered[newly_covered] = False
            # Vectorized gain update
            for s in newly_covered:
                gains[subset_block_arrays[s]] -= 1

        return selected_blocks

    def run_greedy_partial(uncovered_mask, random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()

        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            return []

        gains = np.zeros(num_blocks, dtype=np.int32)
        for s_idx in uncovered_indices:
            gains[subset_block_arrays[s_idx]] += 1

        selected_blocks = []
        current_uncovered = uncovered_mask.copy()

        while True:
            if time.time() > deadline - 25.0:
                break
            if not np.any(current_uncovered):
                break

            noise = rng.uniform(0, 0.3, size=num_blocks)
            scores = gains + noise
            best_idx = int(np.argmax(scores))

            if gains[best_idx] == 0:
                break

            selected_blocks.append(best_idx)

            b_subs = block_subset_arrays[best_idx]
            newly_covered = b_subs[current_uncovered[b_subs]]

            if len(newly_covered) == 0:
                continue

            current_uncovered[newly_covered] = False
            for s in newly_covered:
                gains[subset_block_arrays[s]] -= 1

        return selected_blocks

    def local_search(blocks_indices, time_budget, random_seed=None):
        """Combined deletion and swap local search."""
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()

        current_blocks = list(blocks_indices)

        if not current_blocks:
            return current_blocks

        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            cover_count[block_subset_arrays[b_idx]] += 1

        improved = True
        ls_start = time.time()

        while improved and (time.time() - ls_start) < time_budget:
            improved = False
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)

            for i in indices:
                if i >= len(current_blocks):
                    continue
                if (time.time() - ls_start) > time_budget:
                    break

                b_idx = current_blocks[i]
                b_subs = block_subset_arrays[b_idx]

                # Check if redundant
                is_redundant = True
                for s_idx in b_subs:
                    if cover_count[s_idx] == 1:
                        is_redundant = False
                        break

                if is_redundant:
                    cover_count[b_subs] -= 1
                    current_blocks.pop(i)
                    improved = True
                    break
                else:
                    # Try swap: find a block that covers all unique subsets of this one
                    unique_subs = [s_idx for s_idx in b_subs if cover_count[s_idx] == 1]
                    if len(unique_subs) <= 3:  # Only attempt swap for small unique sets
                        # Find candidate blocks covering all unique subsets
                        candidates = None
                        for s_idx in unique_subs:
                            candidate_set = set(subset_block_arrays[s_idx].tolist())
                            if candidates is None:
                                candidates = candidate_set
                            else:
                                candidates &= candidate_set
                        candidates.discard(b_idx)

                        if candidates:
                            # Pick a random candidate
                            candidate_list = list(candidates)
                            new_block = candidate_list[rng.randrange(len(candidate_list))]

                            # Check if new block is already in current_blocks
                            if new_block not in current_blocks:
                                # Perform swap
                                cover_count[b_subs] -= 1
                                new_subs = block_subset_arrays[new_block]
                                cover_count[new_subs] += 1
                                current_blocks[i] = new_block
                                improved = True
                                break

        return current_blocks

    best_blocks = None
    best_len = float('inf')

    trial = 0
    while time.time() < deadline - 30.0:
        # Strategy 1: Full Greedy + Local Search
        greedy_res = run_greedy_full(trial)
        ls_budget = min(5.0, max(1.0, (deadline - 30.0 - time.time()) * 0.3))
        ls_res = local_search(greedy_res, ls_budget, trial)

        if len(ls_res) < best_len:
            best_len = len(ls_res)
            best_blocks = ls_res

        # Strategy 2: Perturb Best + Repair + Local Search
        if best_blocks is not None and len(best_blocks) > 20 and time.time() < deadline - 35.0:
            perturbed = list(best_blocks)
            num_remove = min(max(3, len(perturbed) // 15), len(perturbed) - 1)
            if num_remove > 0:
                remove_indices = random.sample(range(len(perturbed)), num_remove)
                blocks_to_remove = set(perturbed[i] for i in remove_indices)

                cover_count_temp = np.zeros(num_subsets, dtype=np.int32)
                for b_idx in perturbed:
                    if b_idx not in blocks_to_remove:
                        cover_count_temp[block_subset_arrays[b_idx]] += 1

                uncovered_mask = (cover_count_temp == 0).astype(bool)
                repair_res = run_greedy_partial(uncovered_mask, trial + 1000)

                new_set = [b for b in perturbed if b not in blocks_to_remove]
                new_set.extend(repair_res)
                new_set = list(dict.fromkeys(new_set))

                ls_budget2 = min(4.0, max(1.0, (deadline - 35.0 - time.time()) * 0.25))
                ls_res2 = local_search(new_set, ls_budget2, trial + 2000)

                if len(ls_res2) < best_len:
                    best_len = len(ls_res2)
                    best_blocks = ls_res2

        # Strategy 3: Local search on best (additional polish)
        if best_blocks is not None and len(best_blocks) > 10 and time.time() < deadline - 40.0:
            ls_budget3 = min(3.0, max(0.5, (deadline - 40.0 - time.time()) * 0.2))
            ls_res3 = local_search(best_blocks, ls_budget3, trial + 3000)
            if len(ls_res3) < best_len:
                best_len = len(ls_res3)
                best_blocks = ls_res3

        trial += 1

    # Final polish with remaining time
    if best_blocks is not None and time.time() < deadline - 15.0:
        remaining = deadline - 15.0 - time.time()
        final_res = local_search(best_blocks, remaining, trial + 5000)
        if len(final_res) < best_len:
            best_blocks = final_res

    if best_blocks is None:
        greedy_res = run_greedy_full(0)
        best_blocks = local_search(greedy_res, 5.0, 0)

    return [list(all_blocks[i]) for i in best_blocks]
