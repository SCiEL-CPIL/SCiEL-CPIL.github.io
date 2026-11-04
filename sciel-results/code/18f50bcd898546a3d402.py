import itertools
import numpy as np
import time
import random
from scipy import sparse

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}

    # Precompute all k-blocks and their t-subset indices
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    k_choose_t = len(list(itertools.combinations(range(k), t)))

    block_t_indices = np.empty((n_blocks, k_choose_t), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        for j, subset in enumerate(itertools.combinations(block, t)):
            block_t_indices[i, j] = t_sub_to_idx[subset]

    # Build sparse coverage matrix
    row_indices = np.repeat(np.arange(n_blocks), k_choose_t)
    col_indices = block_t_indices.flatten()
    data = np.ones(n_blocks * k_choose_t, dtype=np.int8)
    cover_matrix = sparse.csr_matrix((data, (row_indices, col_indices)), shape=(n_blocks, n_t))

    # Precompute for each t-subset, the list of blocks that cover it (for fast lookup)
    t_sub_blocks = {}
    for i in range(n_blocks):
        for j in range(k_choose_t):
            idx = block_t_indices[i, j]
            if idx not in t_sub_blocks:
                t_sub_blocks[idx] = []
            t_sub_blocks[idx].append(i)

    def greedy_cover(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        selected = []
        used = np.zeros(n_blocks, dtype=bool)

        while uncovered.any():
            if time.time() > deadline - 5.0:
                break

            gains = (cover_matrix @ uncovered.astype(np.int8)).astype(np.int32)
            gains[used] = -1

            max_gain = gains.max()
            if max_gain <= 0:
                break

            candidates = np.where(gains == max_gain)[0]
            if seed == 0:
                chosen = candidates[0]
            else:
                chosen = candidates[rng.randint(0, len(candidates) - 1)]

            selected.append(chosen)
            used[chosen] = True
            uncovered[block_t_indices[chosen]] = False

        return selected

    def remove_redundant_bitmask(selected):
        """Remove redundant blocks using bitmask-based coverage counting."""
        if not selected:
            return selected

        selected = list(selected)
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b in selected:
            cover_count[block_t_indices[b]] += 1

        changed = True
        while changed and time.time() < deadline - 3.0:
            changed = False
            new_selected = []
            for b in selected:
                sub_idx = block_t_indices[b]
                if np.min(cover_count[sub_idx]) > 1:
                    cover_count[sub_idx] -= 1
                    changed = True
                else:
                    new_selected.append(b)
            selected = new_selected

        return selected

    def one_opt_improve(selected, max_time=None):
        """Try to remove or replace each block to reduce total count."""
        if not selected:
            return selected

        if max_time is None:
            max_time = deadline - 5.0

        selected = list(selected)
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b in selected:
            cover_count[block_t_indices[b]] += 1

        improved = True
        while improved and time.time() < max_time:
            improved = False
            new_selected = []
            for i, b in enumerate(selected):
                if time.time() > max_time:
                    new_selected.append(b)
                    continue

                sub_idx = block_t_indices[b]
                # Can we remove this block?
                if np.min(cover_count[sub_idx]) > 1:
                    cover_count[sub_idx] -= 1
                    improved = True
                    continue
                else:
                    new_selected.append(b)

            selected = new_selected

        return selected

    def local_search_replace(selected, max_rounds=5):
        """Try to replace blocks with better alternatives using 2-opt style swaps."""
        if not selected:
            return selected

        selected = list(selected)
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b in selected:
            cover_count[block_t_indices[b]] += 1

        improved = True
        rnd = 0

        while improved and rnd < max_rounds and time.time() < deadline - 2.0:
            improved = False
            rnd += 1

            # First pass: remove any redundant blocks
            new_selected = []
            for b in selected:
                sub_idx = block_t_indices[b]
                if np.min(cover_count[sub_idx]) > 1:
                    cover_count[sub_idx] -= 1
                    improved = True
                else:
                    new_selected.append(b)
            selected = new_selected

            # Second pass: try 1-opt replacements
            for i in range(len(selected)):
                if time.time() > deadline - 2.0:
                    break

                b = selected[i]
                sub_idx = block_t_indices[b]
                essential = sub_idx[cover_count[sub_idx] == 1]

                if len(essential) == 0:
                    cover_count[sub_idx] -= 1
                    selected.pop(i)
                    improved = True
                    break
                else:
                    # Try to find a replacement that covers all essential subsets
                    # and maximizes net new coverage
                    best_replace = -1
                    best_score = -1

                    # Get candidate blocks that cover the first essential subset
                    first_ess = essential[0]
                    candidates_list = t_sub_blocks.get(first_ess, [])

                    # Limit candidates for speed
                    if len(candidates_list) > 150:
                        rng_local = random.Random(i * 7919 + rnd * 31 + 1)
                        candidates_list = rng_local.sample(candidates_list, 150)

                    essential_set = set(essential)
                    for cand in candidates_list:
                        if cand == b or cand in selected:
                            continue

                        cand_sub = block_t_indices[cand]
                        # Check if cand covers all essential subsets
                        if not essential_set.issubset(set(cand_sub)):
                            continue

                        # Score: count of cand's subsets that are currently uncovered (count==0)
                        # This represents new coverage gained
                        score = np.sum(cover_count[cand_sub] == 0)
                        if score > best_score:
                            best_score = score
                            best_replace = cand

                    if best_replace != -1:
                        old_sub = block_t_indices[b]
                        new_sub = block_t_indices[best_replace]
                        cover_count[old_sub] -= 1
                        cover_count[new_sub] += 1
                        selected[i] = best_replace
                        improved = True
                        break

            # Third pass: try 2-opt (swap two blocks with a pair that gives better coverage)
            if time.time() < deadline - 2.0 and len(selected) >= 2:
                # Sample a few pairs to try
                n_pairs_to_try = min(50, len(selected) * (len(selected) - 1) // 2)
                pair_indices = []
                for i in range(len(selected)):
                    for j in range(i + 1, len(selected)):
                        pair_indices.append((i, j))
                        if len(pair_indices) >= n_pairs_to_try:
                            break
                    if len(pair_indices) >= n_pairs_to_try:
                        break

                rng_pair = random.Random(rnd * 104729 + 13)
                if len(pair_indices) > n_pairs_to_try:
                    pair_indices = rng_pair.sample(pair_indices, n_pairs_to_try)

                for (i, j) in pair_indices:
                    if time.time() > deadline - 2.0:
                        break

                    b1 = selected[i]
                    b2 = selected[j]
                    sub1 = block_t_indices[b1]
                    sub2 = block_t_indices[b2]

                    # Essential subsets: those covered only by b1 or only by b2
                    essential1 = sub1[cover_count[sub1] == 1]
                    essential2 = sub2[cover_count[sub2] == 1]

                    # If either has no essential subsets, we can just remove it (handled above)
                    if len(essential1) == 0 or len(essential2) == 0:
                        continue

                    # Find two new blocks that together cover all essential subsets
                    # and provide better net coverage
                    # Use the first essential subset from each to generate candidates
                    first_ess1 = essential1[0]
                    first_ess2 = essential2[0]

                    cand_list1 = t_sub_blocks.get(first_ess1, [])
                    cand_list2 = t_sub_blocks.get(first_ess2, [])

                    if len(cand_list1) > 50:
                        cand_list1 = random.Random(i * 1009 + j * 13 + rnd).sample(cand_list1, 50)
                    if len(cand_list2) > 50:
                        cand_list2 = random.Random(i * 1009 + j * 13 + rnd + 1).sample(cand_list2, 50)

                    best_pair = None
                    best_pair_score = -1
                    essential1_set = set(essential1)
                    essential2_set = set(essential2)

                    for c1 in cand_list1:
                        if c1 == b1 or c1 == b2 or c1 in selected:
                            continue
                        s1 = block_t_indices[c1]
                        if not essential1_set.issubset(set(s1)):
                            continue
                        for c2 in cand_list2:
                            if c2 == b1 or c2 == b2 or c2 in selected or c2 == c1:
                                continue
                            s2 = block_t_indices[c2]
                            if not essential2_set.issubset(set(s2)):
                                continue

                            # Check that together they cover all essential subsets
                            # (already ensured by individual checks)
                            # Score: new coverage
                            score = np.sum(cover_count[s1] == 0) + np.sum(cover_count[s2] == 0)
                            if score > best_pair_score:
                                best_pair_score = score
                                best_pair = (c1, c2)

                    if best_pair is not None:
                        c1, c2 = best_pair
                        old1 = block_t_indices[b1]
                        old2 = block_t_indices[b2]
                        new1 = block_t_indices[c1]
                        new2 = block_t_indices[c2]
                        cover_count[old1] -= 1
                        cover_count[old2] -= 1
                        cover_count[new1] += 1
                        cover_count[new2] += 1
                        selected[i] = c1
                        selected[j] = c2
                        improved = True
                        break

        return selected

    best_blocks = None
    best_len = float('inf')

    # 1. Deterministic greedy
    selected = greedy_cover(0)
    selected = remove_redundant_bitmask(selected)
    selected = one_opt_improve(selected)
    if selected:
        best_blocks = selected
        best_len = len(selected)

    # 2. Randomized greedy restarts
    n_restarts = 25
    for seed in range(1, n_restarts + 1):
        if time.time() > deadline - 15.0:
            break

        selected = greedy_cover(seed)
        selected = remove_redundant_bitmask(selected)
        selected = one_opt_improve(selected)
        if selected and len(selected) < best_len:
            best_blocks = selected
            best_len = len(selected)

    # 3. Local search on best solution
    if best_blocks and time.time() < deadline - 10.0:
        best_blocks = local_search_replace(best_blocks, max_rounds=5)
        best_blocks = remove_redundant_bitmask(best_blocks)
        best_blocks = one_opt_improve(best_blocks)
        # Run local search again if time permits
        if time.time() < deadline - 5.0:
            best_blocks = local_search_replace(best_blocks, max_rounds=3)
            best_blocks = remove_redundant_bitmask(best_blocks)
            best_blocks = one_opt_improve(best_blocks)

    # 4. Final cleanup
    best_blocks = remove_redundant_bitmask(best_blocks)
    best_blocks = one_opt_improve(best_blocks)

    result_blocks = [list(all_blocks[i]) for i in best_blocks]
    return result_blocks
