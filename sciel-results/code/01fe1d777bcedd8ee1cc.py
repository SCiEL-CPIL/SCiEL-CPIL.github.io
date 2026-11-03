import itertools
import time
import random
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = 118.0  # Safety margin

    def elapsed():
        return time.time() - start_time

    # 1. Precomputation
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_index = {ts: i for i, ts in enumerate(all_tsets)}

    all_blocks = []
    block_masks = []
    for block in itertools.combinations(range(v), k):
        mask = 0
        for ts in itertools.combinations(block, t):
            mask |= (1 << tset_index[ts])
        all_blocks.append(block)
        block_masks.append(mask)

    num_blocks = len(all_blocks)
    full_mask = (1 << num_tsets) - 1

    # Build tset_to_blocks for fast candidate lookup
    tset_to_blocks = [[] for _ in range(num_tsets)]
    for b_idx, mask in enumerate(block_masks):
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            tset_to_blocks[idx].append(b_idx)
            m ^= lsb

    # 2. Weighted Greedy Set Cover with randomization
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        used = [False] * num_blocks
        # Use random weights to diversify greedy choices
        weights = [rng.random() * 0.1 for _ in range(num_blocks)]

        while uncovered:
            if elapsed() > deadline - 12:
                break
            best_idx = -1
            best_score = -1.0

            for i in range(num_blocks):
                if used[i]:
                    continue
                count = (block_masks[i] & uncovered).bit_count()
                if count == 0:
                    continue
                # Score = coverage / (1 + small random perturbation)
                score = count / (1.0 + weights[i])
                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx == -1 or best_score <= 0:
                break

            used[best_idx] = True
            selected.append(best_idx)
            uncovered &= ~block_masks[best_idx]

        return selected

    # 3. Local Search with enhanced moves
    def local_search(initial_sol, seed):
        rng = random.Random(seed)
        sol = list(initial_sol)
        in_sol = [False] * num_blocks
        for b in sol:
            in_sol[b] = True

        cover_counts = [0] * num_tsets

        def update_counts(block_idx, delta):
            m = block_masks[block_idx]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_counts[idx] += delta
                m ^= lsb

        for b in sol:
            update_counts(b, 1)

        best_sol = list(sol)
        best_size = len(sol)
        no_improve = 0
        max_no_improve = 100

        while elapsed() < deadline - 3 and no_improve < max_no_improve:
            improved = False

            # Phase 1: Remove redundant blocks (repeat until none found)
            removed_any = True
            while removed_any and elapsed() < deadline - 3:
                removed_any = False
                order = list(range(len(sol)))
                rng.shuffle(order)
                for pos in order:
                    if pos >= len(sol):
                        continue
                    b = sol[pos]
                    m = block_masks[b]
                    redundant = True
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if cover_counts[idx] == 1:
                            redundant = False
                            break
                        m ^= lsb

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

            # Phase 2: 1-for-1 Swap (more attempts)
            if sol:
                found_swap = False
                num_swap_attempts = 50
                for _ in range(num_swap_attempts):
                    if not sol or elapsed() > deadline - 3:
                        break
                    pos = rng.randint(0, len(sol) - 1)
                    b_out = sol[pos]

                    critical_mask = 0
                    m = block_masks[b_out]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if cover_counts[idx] == 1:
                            critical_mask |= lsb
                        m ^= lsb

                    if critical_mask == 0:
                        update_counts(b_out, -1)
                        in_sol[b_out] = False
                        sol.pop(pos)
                        improved = True
                        found_swap = True
                        break

                    # Find candidate blocks covering all critical t-sets
                    crit_ts_list = []
                    m = critical_mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        crit_ts_list.append(idx)
                        m ^= lsb

                    if not crit_ts_list:
                        continue

                    # Intersect tset_to_blocks for each critical t-set
                    candidates = set(tset_to_blocks[crit_ts_list[0]])
                    for ts_idx in crit_ts_list[1:]:
                        if not candidates:
                            break
                        candidates &= set(tset_to_blocks[ts_idx])

                    # Filter out blocks already in solution
                    candidates = {c for c in candidates if not in_sol[c]}

                    if candidates:
                        b_in = rng.choice(list(candidates))
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

            # Phase 3: 2-for-1 merge (replace 2 blocks with 1 if possible)
            if len(sol) >= 2 and elapsed() < deadline - 3:
                found_merge = False
                for _ in range(20):
                    if elapsed() > deadline - 3:
                        break
                    pos1 = rng.randint(0, len(sol) - 1)
                    pos2 = rng.randint(0, len(sol) - 1)
                    if pos1 == pos2:
                        continue
                    b1 = sol[pos1]
                    b2 = sol[pos2]

                    # Get critical t-sets for both blocks
                    crit_mask = 0
                    for b in [b1, b2]:
                        m = block_masks[b]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            if cover_counts[idx] <= 1:
                                crit_mask |= lsb
                            m ^= lsb

                    if crit_mask == 0:
                        # Both are redundant
                        for pos in sorted([pos1, pos2], reverse=True):
                            b = sol[pos]
                            update_counts(b, -1)
                            in_sol[b] = False
                            sol.pop(pos)
                        improved = True
                        found_merge = True
                        break

                    # Find a single block that covers all critical t-sets
                    crit_ts_list = []
                    m = crit_mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        crit_ts_list.append(idx)
                        m ^= lsb

                    if len(crit_ts_list) > 3:  # Too many constraints, skip
                        continue

                    candidates = set(tset_to_blocks[crit_ts_list[0]])
                    for ts_idx in crit_ts_list[1:]:
                        if not candidates:
                            break
                        candidates &= set(tset_to_blocks[ts_idx])

                    # Filter out blocks already in solution
                    candidates = {c for c in candidates if not in_sol[c]}

                    if candidates:
                        b_in = rng.choice(list(candidates))
                        # Remove both blocks
                        for pos in sorted([pos1, pos2], reverse=True):
                            b = sol[pos]
                            update_counts(b, -1)
                            in_sol[b] = False
                            sol.pop(pos)
                        # Add new block
                        update_counts(b_in, 1)
                        in_sol[b_in] = True
                        sol.append(b_in)
                        found_merge = True
                        improved = True
                        break

                if found_merge:
                    if len(sol) < best_size:
                        best_size = len(sol)
                        best_sol = list(sol)
                    no_improve = 0
                    continue

            no_improve += 1

        return best_sol

    # Main Loop
    best_solution = None
    best_size = float('inf')

    # Run greedy restarts with local search
    num_restarts = 10
    for seed in range(num_restarts):
        if elapsed() > deadline - 15:
            break
        sol = greedy_cover(seed)
        if sol:
            ls_sol = local_search(sol, seed)
            if len(ls_sol) < best_size:
                best_size = len(ls_sol)
                best_solution = ls_sol

    if best_solution is None:
        return []

    return [list(all_blocks[i]) for i in best_solution]
