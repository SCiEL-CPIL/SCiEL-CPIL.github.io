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

    n_t_in_k = len(list(itertools.combinations(range(k), t)))

    # Build the mapping: for each block, which t-subsets does it cover?
    block_t_indices = np.zeros((n_k, n_t_in_k), dtype=np.int32)

    for i, block in enumerate(k_subsets):
        for j, t_combo in enumerate(itertools.combinations(block, t)):
            block_t_indices[i, j] = t_sub_to_idx[t_combo]

    # Precompute reverse mapping: for each t-subset, which k-subsets contain it
    t_to_k = [[] for _ in range(n_t)]
    for i in range(n_k):
        for j in range(n_t_in_k):
            t_idx = block_t_indices[i, j]
            t_to_k[t_idx].append(i)

    # 3. Greedy Set Cover with randomized kicks
    def greedy_cover_with_kicks(num_kicks=0, seed=None):
        if seed is not None:
            random.seed(seed)
        uncovered = np.ones(n_t, dtype=bool)
        selected_blocks = []
        selected_set = set()

        while np.any(uncovered):
            if time.time() - start_time > deadline - safety_margin:
                break

            gains = uncovered[block_t_indices].sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break

            max_mask = gains == max_gain
            candidates = np.where(max_mask)[0]

            if num_kicks > 0 and len(candidates) > 1:
                if random.random() < 0.3:
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
                is_redundant = True
                for t_combo in itertools.combinations(block, t):
                    if coverage_count[t_sub_to_idx[t_combo]] == 1:
                        is_redundant = False
                        break

                if is_redundant:
                    for t_combo in itertools.combinations(block, t):
                        coverage_count[t_sub_to_idx[t_combo]] -= 1
                    selected_blocks.pop(i)
                    changed = True
                    break

        return selected_blocks

    # 5. Local search: try to replace blocks to reduce count
    def local_search(selected_blocks, max_rounds=50):
        if not selected_blocks:
            return selected_blocks

        if time.time() - start_time > deadline - safety_margin:
            return selected_blocks

        coverage_count = np.zeros(n_t, dtype=np.int32)
        for block in selected_blocks:
            for t_combo in itertools.combinations(block, t):
                coverage_count[t_sub_to_idx[t_combo]] += 1

        improved = True
        rounds = 0
        while improved and rounds < max_rounds:
            if time.time() - start_time > deadline - safety_margin:
                break
            improved = False
            rounds += 1

            for i in range(len(selected_blocks)):
                if time.time() - start_time > deadline - safety_margin:
                    break

                block = selected_blocks[i]
                block_t_subs = list(itertools.combinations(block, t))

                # Find t-subsets uniquely covered by this block
                unique_t_subs = []
                for t_combo in block_t_subs:
                    if coverage_count[t_sub_to_idx[t_combo]] == 1:
                        unique_t_subs.append(t_sub_to_idx[t_combo])

                if not unique_t_subs:
                    # Redundant - remove
                    for t_combo in block_t_subs:
                        coverage_count[t_sub_to_idx[t_combo]] -= 1
                    selected_blocks.pop(i)
                    improved = True
                    break

                # Try to find a replacement that covers all unique_t_subs
                if len(unique_t_subs) <= 6:
                    unique_set = set(unique_t_subs)
                    # Gather candidates from first two unique t-subsets
                    cand_set = set()
                    for t_idx in unique_t_subs[:2]:
                        cand_set.update(t_to_k[t_idx])

                    best_cand = None
                    best_add = -1
                    cur_t_set = set(t_sub_to_idx[tc] for tc in block_t_subs)

                    for cand in cand_set:
                        # Skip if it's the same block
                        if tuple(k_subsets[cand]) == tuple(block):
                            continue
                        cand_t = set(block_t_indices[cand].tolist())
                        if unique_set.issubset(cand_t):
                            add = len(cand_t - cur_t_set)
                            if add > best_add:
                                best_add = add
                                best_cand = cand

                    if best_cand is not None:
                        # Perform swap
                        for t_combo in block_t_subs:
                            coverage_count[t_sub_to_idx[t_combo]] -= 1
                        for t_idx in block_t_indices[best_cand]:
                            coverage_count[t_idx] += 1
                        selected_blocks[i] = k_subsets[best_cand]
                        improved = True
                        break

        return selected_blocks

    # 6. Multiple restarts
    best_result = None
    best_score = float('inf')

    # First: deterministic greedy
    result1 = greedy_cover_with_kicks(num_kicks=0)
    result1 = remove_redundant(result1)
    if len(result1) < best_score:
        best_score = len(result1)
        best_result = result1

    # Second: greedy with kicks
    if time.time() - start_time < deadline - safety_margin - 10:
        result2 = greedy_cover_with_kicks(num_kicks=20, seed=42)
        result2 = remove_redundant(result2)
        if len(result2) < best_score:
            best_score = len(result2)
            best_result = result2

    # Third: another restart
    if time.time() - start_time < deadline - safety_margin - 10:
        result3 = greedy_cover_with_kicks(num_kicks=15, seed=123)
        result3 = remove_redundant(result3)
        if len(result3) < best_score:
            best_score = len(result3)
            best_result = result3

    # Fourth: more aggressive kicks
    if time.time() - start_time < deadline - safety_margin - 10:
        result4 = greedy_cover_with_kicks(num_kicks=30, seed=777)
        result4 = remove_redundant(result4)
        if len(result4) < best_score:
            best_score = len(result4)
            best_result = result4

    # Fifth: local search on best result
    if best_result is not None and time.time() - start_time < deadline - safety_margin - 5:
        polished = local_search(list(best_result), max_rounds=30)
        if len(polished) < best_score:
            best_score = len(polished)
            best_result = polished

    # Sixth: another restart
    if time.time() - start_time < deadline - safety_margin - 10:
        result5 = greedy_cover_with_kicks(num_kicks=25, seed=999)
        result5 = remove_redundant(result5)
        if len(result5) < best_score:
            best_score = len(result5)
            best_result = result5
        if time.time() - start_time < deadline - safety_margin - 5:
            polished5 = local_search(list(best_result), max_rounds=20)
            if len(polished5) < best_score:
                best_score = len(polished5)
                best_result = polished5

    # Final: verify and return
    if best_result is None:
        return []

    covered = np.zeros(n_t, dtype=bool)
    for block in best_result:
        for t_combo in itertools.combinations(block, t):
            covered[t_sub_to_idx[t_combo]] = True

    if not np.all(covered):
        result_fallback = greedy_cover_with_kicks(num_kicks=0)
        result_fallback = remove_redundant(result_fallback)
        return [list(b) for b in result_fallback]

    return [list(b) for b in best_result]
