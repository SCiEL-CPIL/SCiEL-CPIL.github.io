import time
import random
import itertools
import numpy as np
import sys
from scipy import sparse

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0

    def C(n, r):
        if r > n or r < 0:
            return 0
        if r == 0 or r == n:
            return 1
        r = min(r, n - r)
        result = 1
        for i in range(1, r + 1):
            result = result * (n - i + 1) // i
        return result

    # 1. Precomputation using bitmasks for speed
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)

    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)

    # Map t-subset (tuple) to index
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}

    # For each block, list of t-subset indices it covers
    block_to_ts = []
    # For each t-subset, list of block indices that cover it
    ts_to_blocks = [[] for _ in range(num_t_subsets)]

    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(np.array(ts_list, dtype=np.int32))

    # Build sparse incidence matrix: rows = blocks, cols = t-subsets
    # M[b, ts] = 1 if block b covers t-subset ts
    row_indices = []
    col_indices = []
    for b_idx, ts_arr in enumerate(block_to_ts):
        for ts_idx in ts_arr:
            row_indices.append(b_idx)
            col_indices.append(ts_idx)
    incidence = sparse.csr_matrix(
        (np.ones(len(row_indices), dtype=np.int8), (row_indices, col_indices)),
        shape=(num_k_blocks, num_t_subsets)
    )

    # Convert ts_to_blocks to numpy arrays for vectorized ops
    ts_to_blocks_arr = [np.array(lst, dtype=np.int32) for lst in ts_to_blocks]

    # Set versions for intersection
    ts_to_blocks_set = [set(lst) for lst in ts_to_blocks]

    # Block tuple to index
    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}

    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)

    # Fast redundancy removal using sparse matrix operations
    def optimize_redundancy_v2(blocks):
        block_indices = []
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                block_indices.append(idx)

        n = len(block_indices)
        if n == 0:
            return []

        # Build cover count using sparse matrix row sum
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for bidx in block_indices:
            cover_count[block_to_ts[bidx]] += 1

        # Iteratively remove blocks with zero unique t-subsets
        while True:
            if time.time() > deadline - 1.0:
                break
            # For each block, count unique t-subsets (those covered exactly once)
            unique_counts = np.zeros(n, dtype=np.int32)
            for i, bidx in enumerate(block_indices):
                ts_arr = block_to_ts[bidx]
                unique_counts[i] = np.sum(cover_count[ts_arr] == 1)

            zero_mask = (unique_counts == 0)
            if not np.any(zero_mask):
                break

            # Remove all zero-unique blocks simultaneously
            zero_positions = np.where(zero_mask)[0]
            for pos in zero_positions:
                bidx = block_indices[pos]
                ts_arr = block_to_ts[bidx]
                cover_count[ts_arr] -= 1

            # Filter out removed blocks
            keep_mask = ~zero_mask
            block_indices = [block_indices[i] for i in range(n) if keep_mask[i]]
            n = len(block_indices)
            if n == 0:
                break

        return [list(all_k_blocks[idx]) for idx in block_indices]

    # Greedy with tie-breaking
    def greedy_deterministic():
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []

        while True:
            if time.time() > deadline - 15.0:
                break
            max_idx = int(np.argmax(gain))
            if gain[max_idx] <= 0:
                break
            sol.append(list(all_k_blocks[max_idx]))
            is_selected[max_idx] = True
            gain[max_idx] = -1
            newly_covered = block_to_ts[max_idx][is_uncovered[block_to_ts[max_idx]]]
            if len(newly_covered) > 0:
                is_uncovered[newly_covered] = False
                for ts_idx in newly_covered:
                    for b_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[b_idx]:
                            gain[b_idx] -= 1
        return sol

    # Randomized greedy with adaptive temperature
    def randomized_greedy(seed, temperature=0.0):
        random.seed(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []
        step = 0

        while True:
            if time.time() > deadline - 10.0:
                return None
            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
            # Adaptive exploration: increase exploration as we go deeper
            explore_prob = min(0.3, 0.05 + 0.02 * (step // 10))
            if temperature > 0 and random.random() < explore_prob:
                explore_gain = max(0, max_gain - 3)
                explore_candidates = np.where((gain >= explore_gain) & (~is_selected))[0]
                if len(explore_candidates) > 0:
                    b_idx = int(explore_candidates[random.randint(0, len(explore_candidates) - 1)])
                else:
                    b_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            else:
                b_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1
            newly_covered = block_to_ts[b_idx][is_uncovered[block_to_ts[b_idx]]]
            if len(newly_covered) > 0:
                is_uncovered[newly_covered] = False
                for ts_idx in newly_covered:
                    for nb_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[nb_idx]:
                            gain[nb_idx] -= 1
            step += 1
        return sol

    # 1-swap local search with better candidate scoring
    def local_search_swap(blocks, max_rounds=30):
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            b_idx = get_block_idx(b)
            if b_idx >= 0:
                cover_count[block_to_ts[b_idx]] += 1

        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)

        improved = False
        for _ in range(max_rounds):
            if time.time() > deadline - 3.0:
                break
            round_improved = False
            # Shuffle order for diversity
            order = list(range(len(blocks)))
            random.shuffle(order)
            for i in order:
                if time.time() > deadline - 3.0:
                    break
                old_b_idx = get_block_idx(blocks[i])
                if old_b_idx < 0:
                    continue
                unique_ts = []
                for ts_idx in block_to_ts[old_b_idx]:
                    if cover_count[ts_idx] == 1:
                        unique_ts.append(ts_idx)
                if not unique_ts:
                    continue
                # Find blocks covering all unique_ts
                candidates = ts_to_blocks_set[unique_ts[0]].copy()
                for ts_idx in unique_ts[1:]:
                    candidates.intersection_update(ts_to_blocks_set[ts_idx])
                    if not candidates:
                        break
                if not candidates:
                    continue
                valid_candidates = [c for c in candidates if c not in sol_block_indices]
                if not valid_candidates:
                    continue
                # Pick the one that maximizes additional coverage
                best_new = None
                best_gain = -1
                for c in valid_candidates[:30]:  # Limit for speed
                    ts_arr = block_to_ts[c]
                    gain_val = np.sum(cover_count[ts_arr] == 0)
                    if gain_val > best_gain:
                        best_gain = gain_val
                        best_new = c
                if best_new is None:
                    best_new = random.choice(valid_candidates)
                new_block_idx = best_new
                new_block = list(all_k_blocks[new_block_idx])
                for ts_idx in block_to_ts[old_b_idx]:
                    cover_count[ts_idx] -= 1
                for ts_idx in block_to_ts[new_block_idx]:
                    cover_count[ts_idx] += 1
                sol_block_indices.discard(old_b_idx)
                sol_block_indices.add(new_block_idx)
                blocks[i] = new_block
                round_improved = True
                improved = True
            if not round_improved:
                break
        return blocks, improved

    # 2-exchange: replace two blocks with one
    def local_search_2exchange(blocks):
        if len(blocks) < 2:
            return blocks, False
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            b_idx = get_block_idx(b)
            if b_idx >= 0:
                cover_count[block_to_ts[b_idx]] += 1
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)
        improved = False
        n = len(blocks)
        # Try multiple random pairs
        for attempt in range(10):
            if time.time() > deadline - 2.0:
                break
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            old_i_idx = get_block_idx(blocks[i])
            old_j_idx = get_block_idx(blocks[j])
            if old_i_idx < 0 or old_j_idx < 0:
                continue
            unique_ts_i = [ts_idx for ts_idx in block_to_ts[old_i_idx] if cover_count[ts_idx] == 1]
            unique_ts_j = [ts_idx for ts_idx in block_to_ts[old_j_idx] if cover_count[ts_idx] == 1]
            if not unique_ts_i and not unique_ts_j:
                continue
            all_unique = unique_ts_i + unique_ts_j
            if not all_unique:
                continue
            candidates = ts_to_blocks_set[all_unique[0]].copy()
            for ts_idx in all_unique[1:]:
                candidates.intersection_update(ts_to_blocks_set[ts_idx])
                if not candidates:
                    break
            if candidates:
                valid = [c for c in candidates if c not in sol_block_indices]
                if valid:
                    new_block_idx = random.choice(valid)
                    new_block = list(all_k_blocks[new_block_idx])
                    for ts_idx in block_to_ts[old_i_idx]:
                        cover_count[ts_idx] -= 1
                    for ts_idx in block_to_ts[old_j_idx]:
                        cover_count[ts_idx] -= 1
                    for ts_idx in block_to_ts[new_block_idx]:
                        cover_count[ts_idx] += 1
                    sol_block_indices.discard(old_i_idx)
                    sol_block_indices.discard(old_j_idx)
                    sol_block_indices.add(new_block_idx)
                    blocks.pop(j)
                    blocks[i] = new_block
                    improved = True
                    break
        return blocks, improved

    # Simulated annealing polish with better move selection
    def sa_polish(blocks, max_iters=5000):
        if not blocks:
            return blocks
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)
                cover_count[block_to_ts[idx]] += 1
        best_len = len(blocks)
        best_blocks = list(blocks)
        current_len = len(blocks)
        T = 50.0
        T_min = 0.5
        cooling = 0.998
        for it in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            uncovered = np.where(cover_count == 0)[0]
            current_obj = len(uncovered) * 1000 + current_len
            if len(uncovered) > 0 and random.random() < 0.5:
                ts_idx = int(uncovered[random.randint(0, len(uncovered) - 1)])
                candidate_blocks = ts_to_blocks[ts_idx]
                if not candidate_blocks:
                    continue
                if random.random() < 0.5 and sol_block_indices:
                    existing_list = list(sol_block_indices)
                    old_idx = existing_list[random.randint(0, len(existing_list) - 1)]
                    valid_candidates = [c for c in candidate_blocks if c not in sol_block_indices]
                    if not valid_candidates:
                        continue
                    new_idx = valid_candidates[random.randint(0, len(valid_candidates) - 1)]
                    old_covered = block_to_ts[old_idx]
                    new_covered = block_to_ts[new_idx]
                    delta_uncovered = 0
                    for ts_idx2 in old_covered:
                        if cover_count[ts_idx2] == 1:
                            delta_uncovered += 1
                    for ts_idx2 in new_covered:
                        if cover_count[ts_idx2] == 0:
                            delta_uncovered -= 1
                    delta_obj = delta_uncovered * 1000
                    if delta_obj <= 0 or random.random() < np.exp(-delta_obj / T):
                        for ts_idx2 in old_covered:
                            cover_count[ts_idx2] -= 1
                        for ts_idx2 in new_covered:
                            cover_count[ts_idx2] += 1
                        sol_block_indices.discard(old_idx)
                        sol_block_indices.add(new_idx)
                else:
                    valid_candidates = [c for c in candidate_blocks if c not in sol_block_indices]
                    if not valid_candidates:
                        continue
                    new_idx = valid_candidates[random.randint(0, len(valid_candidates) - 1)]
                    new_covered = block_to_ts[new_idx]
                    delta_uncovered = 0
                    for ts_idx2 in new_covered:
                        if cover_count[ts_idx2] == 0:
                            delta_uncovered -= 1
                    delta_obj = delta_uncovered * 1000 + 1
                    if delta_obj <= 0 or random.random() < np.exp(-delta_obj / T):
                        for ts_idx2 in new_covered:
                            cover_count[ts_idx2] += 1
                        sol_block_indices.add(new_idx)
                        current_len += 1
            else:
                if not sol_block_indices:
                    continue
                block_criticality = {}
                for b_idx in sol_block_indices:
                    crit = 0
                    for ts_idx2 in block_to_ts[b_idx]:
                        if cover_count[ts_idx2] == 1:
                            crit += 1
                    block_criticality[b_idx] = crit
                sorted_blocks = sorted(block_criticality.items(), key=lambda x: x[1])
                if sorted_blocks[0][1] == 0:
                    old_idx = sorted_blocks[0][0]
                    old_covered = block_to_ts[old_idx]
                    for ts_idx2 in old_covered:
                        cover_count[ts_idx2] -= 1
                    sol_block_indices.discard(old_idx)
                    current_len -= 1
                elif len(sorted_blocks) > 1:
                    old_idx = sorted_blocks[0][0]
                    old_covered = block_to_ts[old_idx]
                    unique_ts = [ts_idx2 for ts_idx2 in old_covered if cover_count[ts_idx2] == 1]
                    if unique_ts:
                        candidates = ts_to_blocks_set[unique_ts[0]].copy()
                        for ts_idx2 in unique_ts[1:]:
                            candidates.intersection_update(ts_to_blocks_set[ts_idx2])
                            if not candidates:
                                break
                        valid = [c for c in candidates if c not in sol_block_indices]
                        if valid:
                            new_idx = random.choice(valid)
                            new_covered = block_to_ts[new_idx]
                            for ts_idx2 in old_covered:
                                cover_count[ts_idx2] -= 1
                            for ts_idx2 in new_covered:
                                cover_count[ts_idx2] += 1
                            sol_block_indices.discard(old_idx)
                            sol_block_indices.add(new_idx)
            # Rebuild and remove redundancy
            blocks_list = [list(all_k_blocks[idx]) for idx in sol_block_indices]
            blocks_list = optimize_redundancy_v2(blocks_list)
            current_len = len(blocks_list)
            # Update cover_count
            cover_count = np.zeros(num_t_subsets, dtype=np.int32)
            sol_block_indices = set()
            for b in blocks_list:
                idx = get_block_idx(b)
                if idx >= 0:
                    sol_block_indices.add(idx)
                    cover_count[block_to_ts[idx]] += 1
            if len(uncovered) == 0 and current_len < best_len:
                best_len = current_len
                best_blocks = list(blocks_list)
            T = max(T_min, T * cooling)
        return best_blocks

    # Main algorithm
    # Phase 1: Deterministic greedy
    best_solution = greedy_deterministic()
    best_len = len(best_solution)
    best_solution = optimize_redundancy_v2(best_solution)
    best_len = len(best_solution)

    # Phase 2: Randomized restarts with varied temperatures
    num_runs = 80
    for i in range(num_runs):
        if time.time() > deadline - 20.0:
            break
        # Vary temperature for diversity
        if i < 20:
            temp = 0.0
        elif i < 40:
            temp = 0.3
        elif i < 60:
            temp = 0.5
        else:
            temp = 0.8
        sol = randomized_greedy(i, temperature=temp)
        if sol is None:
            continue
        sol = optimize_redundancy_v2(sol)
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol

    # Phase 3: Local search with more rounds
    for _ in range(30):
        if time.time() > deadline - 8.0:
            break
        best_solution, improved = local_search_swap(best_solution, max_rounds=15)
        if improved:
            best_solution = optimize_redundancy_v2(best_solution)
            if len(best_solution) < best_len:
                best_len = len(best_solution)
        else:
            best_solution, improved2 = local_search_2exchange(best_solution)
            if improved2:
                best_solution = optimize_redundancy_v2(best_solution)
                if len(best_solution) < best_len:
                    best_len = len(best_solution)
            else:
                break

    # Phase 4: SA polish
    if time.time() < deadline - 5.0:
        polished = sa_polish(best_solution, max_iters=4000)
        polished = optimize_redundancy_v2(polished)
        if len(polished) < best_len:
            best_solution = polished
            best_len = len(best_solution)

    return best_solution
