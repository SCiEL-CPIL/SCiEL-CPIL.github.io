import itertools
import time
import random
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s hard kill

    def elapsed():
        return time.time() - start_time

    # 1. Precomputation
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_index = {ts: i for i, ts in enumerate(all_tsets)}

    all_blocks = []
    block_masks = []
    block_tsets = []  # list of tset indices covered by each block

    for block in itertools.combinations(range(v), k):
        mask = 0
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = tset_index[ts]
            mask |= (1 << idx)
            ts_list.append(idx)
        all_blocks.append(block)
        block_masks.append(mask)
        block_tsets.append(ts_list)

    num_blocks = len(all_blocks)
    full_mask = (1 << num_tsets) - 1

    # Build tset_to_blocks for fast candidate lookup
    tset_to_blocks = [[] for _ in range(num_tsets)]
    for b_idx, ts_list in enumerate(block_tsets):
        for ts_idx in ts_list:
            tset_to_blocks[ts_idx].append(b_idx)

    # 2. Weighted Greedy Set Cover with diversity-aware scoring
    def weighted_greedy_cover(seed, diversity_weight=0.0):
        rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        used = [False] * num_blocks
        
        # Point co-occurrence tracking for diversity
        point_in_blocks = [0] * v
        block_diversity_score = [0.0] * num_blocks

        while uncovered:
            if elapsed() > deadline - 10:
                break
            best_idx = -1
            best_score = -1

            # Compute diversity scores if enabled
            if diversity_weight > 0:
                for i in range(num_blocks):
                    if used[i]:
                        continue
                    # Diversity: prefer blocks that use under-represented points
                    div_score = sum(1.0 / (1 + point_in_blocks[p]) for p in all_blocks[i])
                    block_diversity_score[i] = div_score

            for i in range(num_blocks):
                if used[i]:
                    continue
                count = (block_masks[i] & uncovered).bit_count()
                if count == 0:
                    continue
                
                if diversity_weight > 0:
                    score = count + diversity_weight * block_diversity_score[i]
                else:
                    score = count

                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx == -1 or best_score == 0:
                break

            used[best_idx] = True
            selected.append(best_idx)
            uncovered &= ~block_masks[best_idx]
            
            # Update point co-occurrence
            for p in all_blocks[best_idx]:
                point_in_blocks[p] += 1

        return selected

    # 3. Local Search with 1-for-1, 2-for-1, and 1-for-2 swaps
    def local_search(initial_sol, seed):
        rng = random.Random(seed)
        sol = list(initial_sol)
        in_sol = [False] * num_blocks
        for b in sol:
            in_sol[b] = True

        cover_counts = [0] * num_tsets

        def update_counts(block_idx, delta):
            for ts_idx in block_tsets[block_idx]:
                cover_counts[ts_idx] += delta

        for b in sol:
            update_counts(b, 1)

        best_sol = list(sol)
        best_size = len(sol)
        no_improve = 0
        max_no_improve = 80

        while elapsed() < deadline - 5 and no_improve < max_no_improve:
            improved = False

            # Phase 1: Remove redundant blocks (repeat until none found)
            removed_any = True
            while removed_any and elapsed() < deadline - 5:
                removed_any = False
                order = list(range(len(sol)))
                rng.shuffle(order)
                for pos in order:
                    if pos >= len(sol):
                        continue
                    b = sol[pos]
                    redundant = True
                    for ts_idx in block_tsets[b]:
                        if cover_counts[ts_idx] == 1:
                            redundant = False
                            break

                    if redundant:
                        update_counts(b, -1)
                        in_sol[b] = False
                        sol.pop(pos)
                        removed_any = True
                        if len(sol) < best_size:
                            best_size = len(sol)
                            best_sol = list(sol)
                        break

            if removed_any:
                improved = True
                no_improve = 0
                continue

            # Phase 2: 1-for-1 Swap
            if sol:
                found_swap = False
                max_swap_attempts = 30
                if elapsed() < deadline - 30:
                    max_swap_attempts = 60

                for _ in range(max_swap_attempts):
                    if not sol or elapsed() > deadline - 5:
                        break
                    pos = rng.randint(0, len(sol) - 1)
                    b_out = sol[pos]

                    critical_mask = 0
                    crit_ts_list = []
                    for ts_idx in block_tsets[b_out]:
                        if cover_counts[ts_idx] == 1:
                            critical_mask |= (1 << ts_idx)
                            crit_ts_list.append(ts_idx)

                    if critical_mask == 0:
                        update_counts(b_out, -1)
                        in_sol[b_out] = False
                        sol.pop(pos)
                        improved = True
                        found_swap = True
                        break

                    if not crit_ts_list:
                        continue

                    # Pick the critical tset with the fewest covering blocks
                    min_ts_idx = crit_ts_list[0]
                    min_len = len(tset_to_blocks[min_ts_idx])
                    for ts_idx in crit_ts_list[1:]:
                        curr_len = len(tset_to_blocks[ts_idx])
                        if curr_len < min_len:
                            min_ts_idx = ts_idx
                            min_len = curr_len

                    valid_candidates = []
                    for b_idx in tset_to_blocks[min_ts_idx]:
                        if in_sol[b_idx]:
                            continue
                        if (block_masks[b_idx] & critical_mask) == critical_mask:
                            valid_candidates.append(b_idx)

                    if valid_candidates:
                        b_in = rng.choice(valid_candidates)
                        update_counts(b_out, -1)
                        in_sol[b_out] = False
                        update_counts(b_in, 1)
                        in_sol[b_in] = True
                        sol[pos] = b_in
                        found_swap = True
                        improved = True
                        break

                if found_swap:
                    if len(sol) < best_size:
                        best_size = len(sol)
                        best_sol = list(sol)
                    no_improve = 0
                    continue
            else:
                break

            # Phase 3: 2-for-1 Swap
            if len(sol) >= 2:
                found_21 = False
                max_21_attempts = 20
                if elapsed() < deadline - 30:
                    max_21_attempts = 40

                for _ in range(max_21_attempts):
                    if elapsed() > deadline - 5:
                        break
                    pos1 = rng.randint(0, len(sol) - 1)
                    pos2 = rng.randint(0, len(sol) - 1)
                    if pos1 == pos2:
                        continue

                    b_out1 = sol[pos1]
                    b_out2 = sol[pos2]

                    critical_mask = 0
                    crit_ts_list = []
                    seen = set()
                    for ts_idx in block_tsets[b_out1]:
                        if cover_counts[ts_idx] == 1:
                            critical_mask |= (1 << ts_idx)
                            if ts_idx not in seen:
                                crit_ts_list.append(ts_idx)
                                seen.add(ts_idx)
                    for ts_idx in block_tsets[b_out2]:
                        if cover_counts[ts_idx] == 1:
                            critical_mask |= (1 << ts_idx)
                            if ts_idx not in seen:
                                crit_ts_list.append(ts_idx)
                                seen.add(ts_idx)

                    if critical_mask == 0:
                        update_counts(b_out1, -1)
                        in_sol[b_out1] = False
                        update_counts(b_out2, -1)
                        in_sol[b_out2] = False
                        if pos1 > pos2:
                            pos1, pos2 = pos2, pos1
                        sol.pop(pos2)
                        sol.pop(pos1)
                        improved = True
                        found_21 = True
                        break

                    if not crit_ts_list:
                        continue

                    min_ts_idx = crit_ts_list[0]
                    min_len = len(tset_to_blocks[min_ts_idx])
                    for ts_idx in crit_ts_list[1:]:
                        curr_len = len(tset_to_blocks[ts_idx])
                        if curr_len < min_len:
                            min_ts_idx = ts_idx
                            min_len = curr_len

                    valid_candidates = []
                    for b_idx in tset_to_blocks[min_ts_idx]:
                        if in_sol[b_idx]:
                            continue
                        if (block_masks[b_idx] & critical_mask) == critical_mask:
                            valid_candidates.append(b_idx)

                    if valid_candidates:
                        b_in = rng.choice(valid_candidates)
                        update_counts(b_out1, -1)
                        in_sol[b_out1] = False
                        update_counts(b_out2, -1)
                        in_sol[b_out2] = False
                        if pos1 > pos2:
                            pos1, pos2 = pos2, pos1
                        sol.pop(pos2)
                        sol.pop(pos1)
                        update_counts(b_in, 1)
                        in_sol[b_in] = True
                        sol.append(b_in)
                        found_21 = True
                        improved = True
                        break

                if found_21:
                    if len(sol) < best_size:
                        best_size = len(sol)
                        best_sol = list(sol)
                    no_improve = 0
                    continue

            # Phase 4: 1-for-2 Swap (remove 1 block, add 2 blocks that cover its critical t-sets better)
            if sol and elapsed() < deadline - 15:
                found_12 = False
                max_12_attempts = 10
                if elapsed() < deadline - 40:
                    max_12_attempts = 20

                for _ in range(max_12_attempts):
                    if not sol or elapsed() > deadline - 5:
                        break
                    pos = rng.randint(0, len(sol) - 1)
                    b_out = sol[pos]

                    critical_mask = 0
                    crit_ts_list = []
                    for ts_idx in block_tsets[b_out]:
                        if cover_counts[ts_idx] == 1:
                            critical_mask |= (1 << ts_idx)
                            crit_ts_list.append(ts_idx)

                    if critical_mask == 0:
                        update_counts(b_out, -1)
                        in_sol[b_out] = False
                        sol.pop(pos)
                        improved = True
                        found_12 = True
                        break

                    if not crit_ts_list:
                        continue

                    # Try to find 2 blocks that together cover all critical t-sets
                    # and provide better coverage overall
                    best_pair = None
                    best_pair_score = -1

                    # Sample a few candidate pairs
                    sample_size = min(20, len(tset_to_blocks[crit_ts_list[0]]))
                    candidates = tset_to_blocks[crit_ts_list[0]]
                    sampled = rng.sample(candidates, min(sample_size, len(candidates)))

                    for b1 in sampled:
                        if in_sol[b1]:
                            continue
                        # Check if b1 covers some critical t-sets
                        mask1 = block_masks[b1] & critical_mask
                        if mask1 == 0:
                            continue
                        remaining = critical_mask & ~mask1
                        if remaining == 0:
                            # b1 alone covers all critical t-sets
                            score = (block_masks[b1] & uncovered).bit_count()
                            if score > best_pair_score:
                                best_pair_score = score
                                best_pair = (b1, None)
                            continue
                        # Find b2 that covers the remaining critical t-sets
                        for b2 in sampled:
                            if b2 == b1 or in_sol[b2]:
                                continue
                            mask2 = block_masks[b2] & remaining
                            if mask2 == remaining:
                                # Both together cover all critical t-sets
                                score = ((block_masks[b1] | block_masks[b2]) & uncovered).bit_count()
                                if score > best_pair_score:
                                    best_pair_score = score
                                    best_pair = (b1, b2)

                    if best_pair:
                        b1, b2 = best_pair
                        update_counts(b_out, -1)
                        in_sol[b_out] = False
                        sol.pop(pos)
                        update_counts(b1, 1)
                        in_sol[b1] = True
                        sol.append(b1)
                        if b2 is not None:
                            update_counts(b2, 1)
                            in_sol[b2] = True
                            sol.append(b2)
                        found_12 = True
                        improved = True
                        break

                if found_12:
                    if len(sol) < best_size:
                        best_size = len(sol)
                        best_sol = list(sol)
                    no_improve = 0
                    continue

            no_improve += 1

        return best_sol

    # Main Loop with multiple strategies
    best_solution = None
    best_size = float('inf')

    # Strategy 1: Standard greedy with multiple restarts
    for seed in range(6):
        if elapsed() > deadline - 25:
            break
        sol = weighted_greedy_cover(seed, diversity_weight=0.0)
        if sol:
            ls_sol = local_search(sol, seed)
            if len(ls_sol) < best_size:
                best_size = len(ls_sol)
                best_solution = ls_sol

    # Strategy 2: Diversity-weighted greedy
    for seed in range(4, 10):
        if elapsed() > deadline - 25:
            break
        sol = weighted_greedy_cover(seed, diversity_weight=0.3)
        if sol:
            ls_sol = local_search(sol, seed)
            if len(ls_sol) < best_size:
                best_size = len(ls_sol)
                best_solution = ls_sol

    # Strategy 3: If we have time, do more local search on the best solution
    if best_solution is not None and elapsed() < deadline - 20:
        # Try to improve further with additional local search passes
        improved_sol = local_search(best_solution, 999)
        if len(improved_sol) < best_size:
            best_size = len(improved_sol)
            best_solution = improved_sol

    if best_solution is None:
        return []

    return [list(all_blocks[i]) for i in best_solution]
