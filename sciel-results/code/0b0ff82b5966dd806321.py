import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = 120.0
    safety_margin = 5.0

    # 1. Precompute t-subsets and their indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {subset: i for i, subset in enumerate(t_subsets)}
    n_t = len(t_subsets)

    # 2. Precompute k-subsets and the indices of t-subsets they contain
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)

    # Map each k-subset to a list of t-subset indices
    # C(k, t) is fixed for a given v, k, t
    n_t_in_k = len(list(itertools.combinations(range(k), t)))

    # Build the mapping: for each block, which t-subsets does it cover?
    block_t_indices = np.zeros((n_k, n_t_in_k), dtype=np.int32)

    for i, block in enumerate(k_subsets):
        for j, t_combo in enumerate(itertools.combinations(block, t)):
            block_t_indices[i, j] = t_sub_to_idx[t_combo]

    # 3. Greedy Set Cover with randomized kicks (Grassberger-Scott style)
    def greedy_cover_with_kicks(num_kicks=0, kick_size=3):
        uncovered = np.ones(n_t, dtype=bool)
        selected_blocks = []
        selected_set = set()

        while np.any(uncovered):
            if time.time() - start_time > deadline - safety_margin:
                break

            # Calculate gain for each block
            gains = uncovered[block_t_indices].sum(axis=1)

            max_gain = np.max(gains)
            if max_gain == 0:
                break

            # Find all blocks with max gain
            max_mask = gains == max_gain
            candidates = np.where(max_mask)[0]

            # Apply randomized kick: if we have kicks remaining, perturb selection
            if num_kicks > 0 and len(candidates) > 1:
                # With some probability, pick a non-maximal block to escape local minima
                if random.random() < 0.3:
                    # Pick from top 20% of gains instead of just max
                    threshold = max_gain - 1
                    perturbed_candidates = np.where(gains >= threshold)[0]
                    if len(perturbed_candidates) > 0:
                        candidates = perturbed_candidates
                    num_kicks -= 1

            best_idx = candidates[random.randint(0, len(candidates) - 1)]

            if best_idx not in selected_set:
                selected_blocks.append(k_subsets[best_idx])
                selected_set.add(best_idx)
                uncovered[block_t_indices[best_idx]] = False

        return selected_blocks

    # 4. Remove redundant blocks
    def remove_redundant(selected_blocks):
        if not selected_blocks:
            return selected_blocks

        # Build coverage count
        coverage_count = np.zeros(n_t, dtype=np.int32)
        for block in selected_blocks:
            for t_combo in itertools.combinations(block, t):
                coverage_count[t_sub_to_idx[t_combo]] += 1

        changed = True
        while changed:
            if time.time() - start_time > deadline - safety_margin:
                break
            changed = False
            for i in range(len(selected_blocks)):
                block = selected_blocks[i]
                # Check if redundant: all t-subsets in this block are covered by >= 2 blocks
                is_redundant = True
                for t_combo in itertools.combinations(block, t):
                    if coverage_count[t_sub_to_idx[t_combo]] == 1:
                        is_redundant = False
                        break

                if is_redundant:
                    # Remove block
                    for t_combo in itertools.combinations(block, t):
                        coverage_count[t_sub_to_idx[t_combo]] -= 1
                    selected_blocks.pop(i)
                    changed = True
                    break  # Restart loop as indices changed

        return selected_blocks

    # 5. Multiple restarts with different random seeds and kick configurations
    best_result = None
    best_score = float('inf')

    # First: deterministic greedy (no kicks)
    result1 = greedy_cover_with_kicks(num_kicks=0)
    result1 = remove_redundant(result1)
    if len(result1) < best_score:
        best_score = len(result1)
        best_result = result1

    # Second: greedy with kicks
    if time.time() - start_time < deadline - safety_margin - 10:
        random.seed(42)
        result2 = greedy_cover_with_kicks(num_kicks=20, kick_size=3)
        result2 = remove_redundant(result2)
        if len(result2) < best_score:
            best_score = len(result2)
            best_result = result2

    # Third: another restart with different seed
    if time.time() - start_time < deadline - safety_margin - 10:
        random.seed(123)
        result3 = greedy_cover_with_kicks(num_kicks=15, kick_size=2)
        result3 = remove_redundant(result3)
        if len(result3) < best_score:
            best_score = len(result3)
            best_result = result3

    # Fourth: greedy with more aggressive kicks
    if time.time() - start_time < deadline - safety_margin - 10:
        random.seed(777)
        result4 = greedy_cover_with_kicks(num_kicks=30, kick_size=5)
        result4 = remove_redundant(result4)
        if len(result4) < best_score:
            best_score = len(result4)
            best_result = result4

    # Final: verify and return
    if best_result is None:
        # Fallback: should not happen, but return empty if all failed
        return []

    # Verify coverage (exhaustive check)
    covered = np.zeros(n_t, dtype=bool)
    for block in best_result:
        for t_combo in itertools.combinations(block, t):
            covered[t_sub_to_idx[t_combo]] = True

    if not np.all(covered):
        # Coverage incomplete; fall back to deterministic greedy
        result_fallback = greedy_cover_with_kicks(num_kicks=0)
        result_fallback = remove_redundant(result_fallback)
        return [list(b) for b in result_fallback]

    return [list(b) for b in best_result]
