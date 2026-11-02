import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0

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

    # Precomputation
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

    def run_greedy(seed, explore_prob=0.0):
        random.seed(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []
        
        while True:
            if time.time() > deadline - 5.0:
                return None
            
            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break
                
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
                
            if explore_prob > 0 and random.random() < explore_prob:
                explore_cands = np.where((gain >= max_gain - 3) & (~is_selected))[0]
                if len(explore_cands) > 0:
                    b_idx = int(explore_cands[random.randint(0, len(explore_cands) - 1)])
                else:
                    b_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            else:
                b_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            
            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1
            
            bts = block_to_ts_np[b_idx]
            newly_covered_mask = is_uncovered[bts]
            newly_covered = bts[newly_covered_mask]
            
            if len(newly_covered) == 0:
                continue
                
            is_uncovered[newly_covered] = False
            
            blocks_to_decrement = []
            for ts_idx in newly_covered:
                blocks_to_decrement.extend(ts_to_blocks[ts_idx])
                
            if blocks_to_decrement:
                counts = np.bincount(blocks_to_decrement, minlength=num_k_blocks)
                gain -= counts
                gain[is_selected] = -1000

        return sol

    best_solution = run_greedy(0)
    if best_solution is None:
        best_solution = [list(all_k_blocks[0])]
    best_len = len(best_solution)
    
    # Run multiple randomized greedy with varying exploration
    num_runs = 100
    for i in range(1, num_runs):
        if time.time() > deadline - 40.0:
            break
        
        explore = 0.0 if i < 30 else (0.1 if i < 60 else 0.2)
        sol = run_greedy(i, explore_prob=explore)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol

    def fast_local_search(blocks, max_iters=3000):
        n_blocks = len(blocks)
        if n_blocks == 0:
            return blocks, 0
        
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
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
                    improved_count += 1
                    changed = True
                    break
        
        # Phase B: Try to replace blocks with better ones
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
                sol_block_set.discard(get_block_idx(blocks[i]))
                sol_block_set.add(new_block_idx)
                improved_count += 1
            else:
                np.subtract.at(cover_count, new_ts_arr, 1)
                np.add.at(cover_count, ts_arr_i, 1)
        
        return blocks, improved_count

    def fast_local_search_2exchange(blocks, max_iters=2000):
        n_blocks = len(blocks)
        if n_blocks < 3:
            return blocks, 0
        
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
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
        
        for ts_arr in block_ts_indices:
            np.add.at(cover_count, ts_arr, 1)
        
        sol_block_set = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_set.add(idx)
        
        improved_count = 0
        iters = 0
        
        while iters < max_iters:
            iters += 1
            if time.time() > deadline - 5.0:
                break
            if len(blocks) < 3:
                break
            
            i = random.randint(0, len(blocks) - 1)
            j = random.randint(0, len(blocks) - 1)
            if i == j:
                continue
            
            ts_i = block_ts_indices[i]
            ts_j = block_ts_indices[j]
            
            unique_i_mask = (cover_count[ts_i] == 1)
            unique_i = ts_i[unique_i_mask]
            unique_j_mask = (cover_count[ts_j] == 1)
            unique_j = ts_j[unique_j_mask]
            
            all_unique = np.concatenate([unique_i, unique_j]) if (len(unique_i) > 0 or len(unique_j) > 0) else np.array([], dtype=np.int32)
            
            if len(all_unique) == 0:
                np.subtract.at(cover_count, ts_i, 1)
                blocks.pop(i)
                block_ts_indices.pop(i)
                if i < len(blocks):
                    j = j - 1 if j > i else j
                improved_count += 1
                continue
            
            candidates = set(ts_to_blocks[int(all_unique[0])])
            for ts_idx in all_unique[1:]:
                candidates.intersection_update(ts_to_blocks[int(ts_idx)])
                if not candidates:
                    break
            
            idx_i = get_block_idx(blocks[i])
            idx_j = get_block_idx(blocks[j])
            candidates.discard(idx_i)
            candidates.discard(idx_j)
            
            if not candidates:
                continue
            
            new_block_idx = random.choice(list(candidates))
            new_ts_arr = block_to_ts_np[new_block_idx]
            
            np.subtract.at(cover_count, ts_i, 1)
            np.subtract.at(cover_count, ts_j, 1)
            np.add.at(cover_count, new_ts_arr, 1)
            
            if np.min(cover_count) >= 1:
                hi = max(i, j)
                lo = min(i, j)
                blocks.pop(hi)
                block_ts_indices.pop(hi)
                blocks.pop(lo)
                block_ts_indices.pop(lo)
                
                blocks.append(list(all_k_blocks[new_block_idx]))
                block_ts_indices.append(new_ts_arr)
                sol_block_set.discard(idx_i)
                sol_block_set.discard(idx_j)
                sol_block_set.add(new_block_idx)
                improved_count += 1
            else:
                np.subtract.at(cover_count, new_ts_arr, 1)
                np.add.at(cover_count, ts_j, 1)
                np.add.at(cover_count, ts_i, 1)
        
        return blocks, improved_count

    # Phase 2: Iterative local search
    for _ in range(150):
        if time.time() > deadline - 10.0:
            break
        best_solution, improved = fast_local_search(best_solution, max_iters=2000)
        if len(best_solution) < best_len:
            best_len = len(best_solution)
        
        if time.time() > deadline - 8.0:
            break
        best_solution, improved2 = fast_local_search_2exchange(best_solution, max_iters=1500)
        if len(best_solution) < best_len:
            best_len = len(best_solution)
        
        total_improved = improved + improved2
        if total_improved == 0 and time.time() > deadline - 20.0:
            break

    # Final polish: aggressive redundancy removal
    for _ in range(30):
        if time.time() > deadline - 2.0:
            break
        changed = False
        for i in range(len(best_solution)):
            if time.time() > deadline - 2.0:
                break
            b = best_solution[i]
            idx = get_block_idx(b)
            if idx < 0:
                continue
            ts_arr = block_to_ts_np[idx]
            
            temp_cover = np.zeros(num_t_subsets, dtype=np.int32)
            for j, b2 in enumerate(best_solution):
                if j == i:
                    continue
                idx2 = get_block_idx(b2)
                if idx2 >= 0:
                    np.add.at(temp_cover, block_to_ts_np[idx2], 1)
            
            if np.all(temp_cover[ts_arr] >= 1):
                best_solution.pop(i)
                if len(best_solution) < best_len:
                    best_len = len(best_solution)
                changed = True
                break
        if not changed:
            break

    return best_solution
