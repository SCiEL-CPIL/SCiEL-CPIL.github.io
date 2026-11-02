import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 142.0

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

    # Precompute all t-subsets and k-blocks
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
        block_to_ts.append(ts_list)

    block_to_ts_np = [np.array(lst, dtype=np.int32) for lst in block_to_ts]
    ts_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in ts_to_blocks]

    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}

    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)

    # Density-driven block generation: for a given set of uncovered t-subsets,
    # find the k-block that covers the most of them.
    def find_best_block(uncovered_mask, seed=0):
        """
        Find the k-block with maximum overlap with uncovered t-subsets.
        Uses a greedy construction heuristic.
        """
        rng = random.Random(seed)
        
        # Start with empty block, greedily add points
        # Score of a partial block = number of uncovered t-subsets it could still cover
        # We use a simpler approach: try to maximize immediate coverage
        
        # For efficiency, we'll use a limited search
        # First, find which points appear in the most uncovered t-subsets
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            return None
        
        # Count point frequencies in uncovered t-subsets
        point_freq = np.zeros(v, dtype=np.int32)
        for ts_idx in uncovered_indices:
            for p in all_t_subsets[ts_idx]:
                point_freq[p] += 1
        
        # Greedily select k points
        selected = []
        remaining = list(range(v))
        
        for _ in range(k):
            if not remaining:
                break
            # Score each candidate point by how many uncovered t-subsets it helps
            best_score = -1
            best_point = None
            for p in remaining:
                # Approximate: how many uncovered t-subsets contain p
                # and could be completed with k-len(selected)-1 more points
                score = point_freq[p]
                if score > best_score:
                    best_score = score
                    best_point = p
            if best_point is None:
                break
            selected.append(best_point)
            remaining.remove(best_point)
        
        if len(selected) < k:
            # Fill with random
            for p in remaining:
                if len(selected) >= k:
                    break
                selected.append(p)
        
        if len(selected) < k:
            return None
        
        selected.sort()
        return tuple(selected)

    def run_greedy_density(seed, max_iters=100000):
        """
        Density-driven greedy: at each step, find the block that covers the most
        currently uncovered t-subsets.
        """
        rng = random.Random(seed)
        is_uncovered = np.ones(num_t_subsets, dtype=bool)
        is_selected = np.zeros(num_k_blocks, dtype=bool)
        sol = []
        sol_block_set = set()
        
        # Track gains: for each block, how many uncovered t-subsets it covers
        gain = np.zeros(num_k_blocks, dtype=np.int32)
        for j in range(num_k_blocks):
            gain[j] = np.sum(is_uncovered[block_to_ts_np[j]])
        
        iters = 0
        while iters < max_iters:
            iters += 1
            if time.time() > deadline - 5.0:
                return sol
            
            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break
            
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
            
            # With some probability, try a density-driven block
            if rng.random() < 0.3:
                # Try to find a better block via density-driven construction
                b_idx = int(candidates[rng.randint(0, len(candidates) - 1)])
                current_block = all_k_blocks[b_idx]
                
                # Try to find a block that covers more uncovered t-subsets
                # by modifying the current block
                improved = False
                for attempt in range(5):
                    # Randomly swap one element
                    new_block = list(current_block)
                    pos = rng.randint(0, k - 1)
                    old_elem = new_block[pos]
                    new_elem = rng.randint(0, v - 1)
                    if new_elem == old_elem:
                        continue
                    if new_elem in new_block:
                        continue
                    new_block[pos] = new_elem
                    new_block.sort()
                    new_tuple = tuple(new_block)
                    new_idx = get_block_idx(new_tuple)
                    if new_idx < 0 or is_selected[new_idx]:
                        continue
                    new_gain_val = int(np.sum(is_uncovered[block_to_ts_np[new_idx]]))
                    if new_gain_val > max_gain:
                        b_idx = new_idx
                        current_block = new_tuple
                        max_gain = new_gain_val
                        improved = True
                        break
                    elif new_gain_val == max_gain:
                        b_idx = new_idx
                        current_block = new_tuple
            
            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1
            sol_block_set.add(b_idx)
            
            bts = block_to_ts_np[b_idx]
            newly_covered_mask = is_uncovered[bts]
            newly_covered = bts[newly_covered_mask]
            
            if len(newly_covered) == 0:
                continue
            
            is_uncovered[newly_covered] = False
            
            # Update gains
            all_blocks_to_decrement = []
            for ts_idx in newly_covered:
                all_blocks_to_decrement.extend(ts_to_blocks[ts_idx])
            if all_blocks_to_decrement:
                counts = np.bincount(all_blocks_to_decrement, minlength=num_k_blocks)
                gain -= counts
                gain[is_selected] = -1000
            
            # Check if all covered
            if not np.any(is_uncovered):
                break
        
        return sol

    def local_search(blocks, max_iters=5000):
        """
        Local search: try to remove redundant blocks and swap blocks for better ones.
        """
        n_blocks = len(blocks)
        if n_blocks == 0:
            return blocks, 0

        block_ts_indices = []
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                block_ts_indices.append(block_to_ts_np[idx])
            else:
                ts_list = []
                for ts in itertools.combinations(b, t):
                    ts_list.append(ts_index_map[ts])
                arr = np.array(ts_list, dtype=np.int32)
                block_ts_indices.append(arr)

        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for ts_arr in block_ts_indices:
            np.add.at(cover_count, ts_arr, 1)

        sol_block_set = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_set.add(idx)

        improved_count = 0

        # Phase A: Remove redundant blocks
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 3.0:
                    return blocks, improved_count
                ts_arr = block_ts_indices[i]
                if np.all(cover_count[ts_arr] >= 2):
                    np.subtract.at(cover_count, ts_arr, 1)
                    blocks.pop(i)
                    block_ts_indices.pop(i)
                    if i < len(blocks):
                        idx = get_block_idx(blocks[i])
                    else:
                        idx = -1
                    if idx >= 0:
                        sol_block_set.discard(idx)
                    improved_count += 1
                    changed = True
                    break

        # Phase B: 1-exchange local search
        iters = 0
        while iters < max_iters:
            iters += 1
            if time.time() > deadline - 3.0:
                break
            if len(blocks) == 0:
                break

            i = random.randint(0, len(blocks) - 1)
            ts_arr_i = block_ts_indices[i]

            unique_mask = cover_count[ts_arr_i] == 1
            unique_ts = ts_arr_i[unique_mask]

            if len(unique_ts) == 0:
                np.subtract.at(cover_count, ts_arr_i, 1)
                blocks.pop(i)
                block_ts_indices.pop(i)
                improved_count += 1
                continue

            candidates = set(ts_to_blocks[int(unique_ts[0])])
            for ts_idx in unique_ts[1:]:
                candidates.intersection_update(ts_to_blocks[int(ts_idx)])
                if not candidates:
                    break

            if not candidates:
                continue

            valid_candidates = [c for c in candidates if c not in sol_block_set]
            if not valid_candidates:
                continue

            new_block_idx = random.choice(valid_candidates)
            new_ts_arr = block_to_ts_np[new_block_idx]

            np.subtract.at(cover_count, ts_arr_i, 1)
            np.add.at(cover_count, new_ts_arr, 1)

            if np.min(cover_count) >= 1:
                blocks[i] = list(all_k_blocks[new_block_idx])
                block_ts_indices[i] = new_ts_arr
                old_idx = get_block_idx(blocks[i])
                if old_idx >= 0:
                    sol_block_set.discard(old_idx)
                sol_block_set.add(new_block_idx)
                improved_count += 1
            else:
                np.subtract.at(cover_count, new_ts_arr, 1)
                np.add.at(cover_count, ts_arr_i, 1)

        return blocks, improved_count

    # Initial deterministic greedy
    best_solution = run_greedy_density(0)
    if best_solution is None or len(best_solution) == 0:
        # Fallback to standard greedy
        best_solution = [list(all_k_blocks[0])]
    best_len = len(best_solution)

    # Run multiple density-driven greedy with varying seeds
    num_runs = 80
    for i in range(1, num_runs):
        if time.time() > deadline - 40.0:
            break
        sol = run_greedy_density(i)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol

    # Phase 2: Iterative local search
    for _ in range(200):
        if time.time() > deadline - 5.0:
            break
        best_solution, improved = local_search(best_solution, max_iters=3000)
        if len(best_solution) < best_len:
            best_len = len(best_solution)
        if improved == 0 and time.time() > deadline - 15.0:
            break

    return best_solution
