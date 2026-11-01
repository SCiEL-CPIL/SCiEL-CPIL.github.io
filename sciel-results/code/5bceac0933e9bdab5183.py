import time
import random
import itertools
import numpy as np
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

    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)

    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)

    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}

    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]

    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(np.array(ts_list, dtype=np.int32))

    ts_to_blocks_arr = [np.array(lst, dtype=np.int32) for lst in ts_to_blocks]
    ts_to_blocks_set = [set(lst) for lst in ts_to_blocks]
    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}

    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)

    def optimize_redundancy(blocks):
        block_indices = []
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                block_indices.append(idx)
        n = len(block_indices)
        if n == 0:
            return []
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for bidx in block_indices:
            cover_count[block_to_ts[bidx]] += 1
        # Multiple passes to remove redundant blocks
        for pass_num in range(5):
            if time.time() > deadline - 1.0:
                break
            unique_counts = np.zeros(n, dtype=np.int32)
            for i, bidx in enumerate(block_indices):
                ts_arr = block_to_ts[bidx]
                unique_counts[i] = np.sum(cover_count[ts_arr] == 1)
            zero_mask = (unique_counts == 0)
            if not np.any(zero_mask):
                break
            zero_positions = np.where(zero_mask)[0]
            for pos in zero_positions:
                bidx = block_indices[pos]
                cover_count[block_to_ts[bidx]] -= 1
            keep_mask = ~zero_mask
            block_indices = [block_indices[i] for i in range(n) if keep_mask[i]]
            n = len(block_indices)
            if n == 0:
                break
        return [list(all_k_blocks[idx]) for idx in block_indices]

    def greedy(seed=0, explore=0.0):
        random.seed(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []
        step = 0
        while True:
            if time.time() > deadline - 8.0:
                return None
            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break
            if explore > 0 and random.random() < explore:
                min_gain = max(0, max_gain - 2)
                candidates = np.where((gain >= min_gain) & (~is_selected))[0]
            else:
                candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
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

    def local_search_1swap(blocks, max_rounds=20):
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)
                cover_count[block_to_ts[idx]] += 1
        improved = False
        for _ in range(max_rounds):
            if time.time() > deadline - 3.0:
                break
            round_improved = False
            order = list(range(len(blocks)))
            random.shuffle(order)
            for i in order:
                if time.time() > deadline - 3.0:
                    break
                old_b_idx = get_block_idx(blocks[i])
                if old_b_idx < 0:
                    continue
                unique_ts = [ts_idx for ts_idx in block_to_ts[old_b_idx] if cover_count[ts_idx] == 1]
                if not unique_ts:
                    continue
                # Find candidates that cover all unique t-subsets of this block
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
                # Evaluate multiple candidates and pick the best
                best_new = None
                best_gain = -1
                for c in valid_candidates[:30]:  # Evaluate more candidates
                    ts_arr = block_to_ts[c]
                    # Gain = number of currently uncovered t-subsets this block covers
                    gain_val = np.sum(cover_count[ts_arr] == 0)
                    # Penalty = number of t-subsets that would become uncovered if we remove old block
                    # but are covered by this new block (i.e., they're in both)
                    if gain_val > best_gain:
                        best_gain = gain_val
                        best_new = c
                if best_new is None:
                    best_new = random.choice(valid_candidates)
                # Verify the swap is valid: new block must cover all unique t-subsets
                new_ts_arr = block_to_ts[best_new]
                all_covered = True
                for ts_idx in unique_ts:
                    if cover_count[ts_idx] == 1 and ts_idx not in new_ts_arr:
                        all_covered = False
                        break
                if not all_covered:
                    continue
                # Perform the swap
                for ts_idx in block_to_ts[old_b_idx]:
                    cover_count[ts_idx] -= 1
                for ts_idx in block_to_ts[best_new]:
                    cover_count[ts_idx] += 1
                sol_block_indices.discard(old_b_idx)
                sol_block_indices.add(best_new)
                blocks[i] = list(all_k_blocks[best_new])
                round_improved = True
                improved = True
            if not round_improved:
                break
        return blocks, improved

    def local_search_2exchange(blocks):
        if len(blocks) < 2:
            return blocks, False
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)
                cover_count[block_to_ts[idx]] += 1
        improved = False
        n = len(blocks)
        for attempt in range(30):  # More attempts
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
            # Find blocks that cover all unique t-subsets from both blocks
            candidates = ts_to_blocks_set[all_unique[0]].copy()
            for ts_idx in all_unique[1:]:
                candidates.intersection_update(ts_to_blocks_set[ts_idx])
                if not candidates:
                    break
            if candidates:
                valid = [c for c in candidates if c not in sol_block_indices]
                if valid:
                    # Pick the best candidate (most additional coverage)
                    best_new = None
                    best_gain = -1
                    for c in valid[:20]:
                        ts_arr = block_to_ts[c]
                        gain_val = np.sum(cover_count[ts_arr] == 0)
                        if gain_val > best_gain:
                            best_gain = gain_val
                            best_new = c
                    if best_new is None:
                        best_new = random.choice(valid)
                    # Perform the 2-exchange
                    for ts_idx in block_to_ts[old_i_idx]:
                        cover_count[ts_idx] -= 1
                    for ts_idx in block_to_ts[old_j_idx]:
                        cover_count[ts_idx] -= 1
                    for ts_idx in block_to_ts[best_new]:
                        cover_count[ts_idx] += 1
                    sol_block_indices.discard(old_i_idx)
                    sol_block_indices.discard(old_j_idx)
                    sol_block_indices.add(best_new)
                    # Remove the later index first to preserve indices
                    if j > i:
                        blocks.pop(j)
                        blocks[i] = list(all_k_blocks[best_new])
                    else:
                        blocks.pop(i)
                        blocks[j] = list(all_k_blocks[best_new])
                    n -= 1
                    improved = True
                    break
        return blocks, improved

    # Phase 1: Deterministic greedy
    best_solution = greedy(seed=0, explore=0.0)
    best_len = len(best_solution)
    best_solution = optimize_redundancy(best_solution)
    best_len = len(best_solution)

    # Phase 2: Randomized restarts
    num_runs = 100
    for i in range(num_runs):
        if time.time() > deadline - 25.0:
            break
        explore = 0.0 if i < 30 else (0.1 if i < 60 else 0.2)
        sol = greedy(seed=i, explore=explore)
        if sol is None:
            continue
        sol = optimize_redundancy(sol)
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol

    # Phase 3: Local search - more iterations
    for _ in range(60):
        if time.time() > deadline - 8.0:
            break
        best_solution, improved = local_search_1swap(best_solution, max_rounds=15)
        if improved:
            best_solution = optimize_redundancy(best_solution)
            if len(best_solution) < best_len:
                best_len = len(best_solution)
        else:
            best_solution, improved2 = local_search_2exchange(best_solution)
            if improved2:
                best_solution = optimize_redundancy(best_solution)
                if len(best_solution) < best_len:
                    best_len = len(best_solution)
            else:
                # Try a few more 1-swaps before giving up
                for _inner in range(3):
                    if time.time() > deadline - 8.0:
                        break
                    best_solution, improved3 = local_search_1swap(best_solution, max_rounds=10)
                    if improved3:
                        best_solution = optimize_redundancy(best_solution)
                        if len(best_solution) < best_len:
                            best_len = len(best_solution)
                        break
                else:
                    break

    return best_solution
