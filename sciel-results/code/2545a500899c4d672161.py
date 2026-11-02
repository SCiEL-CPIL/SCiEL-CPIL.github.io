import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 146.0  # Safety margin for 150s limit

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

    # Precompute as numpy arrays
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

    # Initial deterministic greedy
    best_solution = run_greedy(0)
    if best_solution is None:
        best_solution = [list(all_k_blocks[0])]
    best_len = len(best_solution)
    
    # Run multiple randomized greedy with varying exploration
    num_runs = 120
    for i in range(1, num_runs):
        if time.time() > deadline - 30.0:
            break
        
        explore = 0.0 if i < 40 else (0.1 if i < 80 else 0.2)
        sol = run_greedy(i, explore_prob=explore)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol

    # Fast local search using bitset-like numpy operations
    def fast_local_search(blocks, max_iters=5000):
        n_blocks = len(blocks)
        if n_blocks == 0:
            return blocks, 0
        
        # Build cover count
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
        
        # Track which blocks are in solution
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
                # Check if all t-subsets in this block are covered by >= 2 blocks
                if np.all(cover_count[ts_arr] >= 2):
                    # Remove block i
                    np.subtract.at(cover_count, ts_arr, 1)
                    blocks.pop(i)
                    block_ts_indices.pop(i)
                    idx = get_block_idx(blocks[i] if i < len(blocks) else [0]*k)
                    if idx >= 0:
                        sol_block_set.discard(idx)
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
            
            # Find t-subsets uniquely covered by block i
            unique_mask = cover_count[ts_arr_i] == 1
            unique_ts = ts_arr_i[unique_mask]
            
            if len(unique_ts) == 0:
                # Block is redundant, remove it
                np.subtract.at(cover_count, ts_arr_i, 1)
                blocks.pop(i)
                block_ts_indices.pop(i)
                improved_count += 1
                continue
            
            # Find candidate blocks that cover all unique_ts
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
            
            # Score candidates by overlap with existing coverage (prefer high overlap)
            if len(valid_candidates) > 1:
                scored = []
                for c in valid_candidates:
                    new_ts = block_to_ts_np[c]
                    score = int(np.sum(cover_count[new_ts] >= 2))
                    scored.append((score, c))
                scored.sort(key=lambda x: -x[0])
                # Pick from top candidates with some randomness
                top_n = min(5, len(scored))
                new_block_idx = scored[random.randint(0, top_n - 1)][1]
            else:
                new_block_idx = valid_candidates[0]
            new_ts_arr = block_to_ts_np[new_block_idx]
            
            # Check if swap is valid: after removing block i and adding new_block,
            # all t-subsets still covered
            # Temporarily remove block i's coverage
            np.subtract.at(cover_count, ts_arr_i, 1)
            # Add new block's coverage
            np.add.at(cover_count, new_ts_arr, 1)
            
            if np.min(cover_count) >= 1:
                # Swap is valid
                blocks[i] = list(all_k_blocks[new_block_idx])
                block_ts_indices[i] = new_ts_arr
                sol_block_set.discard(get_block_idx(blocks[i]))
                sol_block_set.add(new_block_idx)
                improved_count += 1
            else:
                # Revert
                np.subtract.at(cover_count, new_ts_arr, 1)
                np.add.at(cover_count, ts_arr_i, 1)
        
        return blocks, improved_count

    def fast_local_search_2exchange(blocks, max_iters=3000):
        """Try to remove 2 blocks and add 2 different blocks (or remove 2, add 1)."""
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
            
            # Try 2-for-1: remove 2 blocks, add 1 block
            i = random.randint(0, len(blocks) - 1)
            j = random.randint(0, len(blocks) - 1)
            if i == j:
                continue
            
            ts_i = block_ts_indices[i]
            ts_j = block_ts_indices[j]
            
            # Unique t-subsets of block i (covered only by i among all blocks)
            unique_i_mask = (cover_count[ts_i] == 1)
            unique_i = ts_i[unique_i_mask]
            
            # Unique t-subsets of block j
            unique_j_mask = (cover_count[ts_j] == 1)
            unique_j = ts_j[unique_j_mask]
            
            # For 2-for-1: we need a single block that covers all unique_i and unique_j
            all_unique = np.concatenate([unique_i, unique_j]) if len(unique_i) > 0 or len(unique_j) > 0 else np.array([], dtype=np.int32)
            
            if len(all_unique) == 0:
                # Both blocks are redundant, remove one
                np.subtract.at(cover_count, ts_i, 1)
                blocks.pop(i)
                block_ts_indices.pop(i)
                if i < len(blocks):
                    j = j - 1 if j > i else j
                improved_count += 1
                continue
            
            # Find blocks that cover all unique t-subsets
            candidates = set(ts_to_blocks[int(all_unique[0])])
            for ts_idx in all_unique[1:]:
                candidates.intersection_update(ts_to_blocks[int(ts_idx)])
                if not candidates:
                    break
            
            # Exclude blocks i and j themselves
            idx_i = get_block_idx(blocks[i])
            idx_j = get_block_idx(blocks[j])
            candidates.discard(idx_i)
            candidates.discard(idx_j)
            
            if not candidates:
                continue
            
            new_block_idx = random.choice(list(candidates))
            new_ts_arr = block_to_ts_np[new_block_idx]
            
            # Check: after removing blocks i and j, adding new_block, all covered
            np.subtract.at(cover_count, ts_i, 1)
            np.subtract.at(cover_count, ts_j, 1)
            np.add.at(cover_count, new_ts_arr, 1)
            
            if np.min(cover_count) >= 1:
                # Valid 2-for-1 swap
                # Remove blocks i and j (higher index first)
                hi = max(i, j)
                lo = min(i, j)
                blocks.pop(hi)
                block_ts_indices.pop(hi)
                blocks.pop(lo)
                block_ts_indices.pop(lo)
                
                # Add new block
                blocks.append(list(all_k_blocks[new_block_idx]))
                block_ts_indices.append(new_ts_arr)
                sol_block_set.discard(idx_i)
                sol_block_set.discard(idx_j)
                sol_block_set.add(new_block_idx)
                improved_count += 1
            else:
                # Revert
                np.subtract.at(cover_count, new_ts_arr, 1)
                np.add.at(cover_count, ts_j, 1)
                np.add.at(cover_count, ts_i, 1)
        
        return blocks, improved_count

    def fast_local_search_2for2(blocks, max_iters=2000):
        """Try to replace 2 blocks with 2 different blocks."""
        n_blocks = len(blocks)
        if n_blocks < 4:
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
            if len(blocks) < 4:
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
                # Both redundant, remove one
                np.subtract.at(cover_count, ts_i, 1)
                blocks.pop(i)
                block_ts_indices.pop(i)
                improved_count += 1
                continue
            
            # Find candidate blocks that cover all unique t-subsets
            candidates = set(ts_to_blocks[int(all_unique[0])])
            for ts_idx in all_unique[1:]:
                candidates.intersection_update(ts_to_blocks[int(ts_idx)])
                if not candidates:
                    break
            
            idx_i = get_block_idx(blocks[i])
            idx_j = get_block_idx(blocks[j])
            candidates.discard(idx_i)
            candidates.discard(idx_j)
            
            if len(candidates) < 2:
                continue
            
            # Try to find 2 candidate blocks that together cover all unique t-subsets
            cand_list = list(candidates)
            random.shuffle(cand_list)
            
            found = False
            for a in range(min(5, len(cand_list))):
                for b_cand in range(a + 1, min(5, len(cand_list))):
                    new_ts_a = block_to_ts_np[cand_list[a]]
                    new_ts_b = block_to_ts_np[cand_list[b_cand]]
                    
                    # Check if new_ts_a and new_ts_b together cover all unique
                    # Build a coverage check
                    temp_cover = cover_count.copy()
                    np.subtract.at(temp_cover, ts_i, 1)
                    np.subtract.at(temp_cover, ts_j, 1)
                    np.add.at(temp_cover, new_ts_a, 1)
                    np.add.at(temp_cover, new_ts_b, 1)
                    
                    if np.min(temp_cover) >= 1:
                        # Valid 2-for-2 swap
                        hi = max(i, j)
                        lo = min(i, j)
                        blocks.pop(hi)
                        block_ts_indices.pop(hi)
                        blocks.pop(lo)
                        block_ts_indices.pop(lo)
                        
                        blocks.append(list(all_k_blocks[cand_list[a]]))
                        block_ts_indices.append(new_ts_a)
                        blocks.append(list(all_k_blocks[cand_list[b_cand]]))
                        block_ts_indices.append(new_ts_b)
                        
                        sol_block_set.discard(idx_i)
                        sol_block_set.discard(idx_j)
                        sol_block_set.add(cand_list[a])
                        sol_block_set.add(cand_list[b_cand])
                        
                        improved_count += 1
                        found = True
                        break
                if found:
                    break
        
        return blocks, improved_count

    # Phase 2: Iterative local search with 1-exchange, 2-for-1, and 2-for-2
    for _ in range(300):
        if time.time() > deadline - 5.0:
            break
        best_solution, improved = fast_local_search(best_solution, max_iters=2000)
        if len(best_solution) < best_len:
            best_len = len(best_solution)
        
        if time.time() > deadline - 8.0:
            break
        best_solution, improved2 = fast_local_search_2exchange(best_solution, max_iters=1500)
        if len(best_solution) < best_len:
            best_len = len(best_solution)
        
        if time.time() > deadline - 5.0:
            break
        best_solution, improved3 = fast_local_search_2for2(best_solution, max_iters=800)
        if len(best_solution) < best_len:
            best_len = len(best_solution)
        
        total_improved = improved + improved2 + improved3
        if total_improved == 0:
            # Try a few more rounds before giving up
            if time.time() > deadline - 10.0:
                break

    # Final polish: aggressive redundancy removal using vectorized cover_count
    for _ in range(100):
        if time.time() > deadline - 2.0:
            break
        if len(best_solution) == 0:
            break
        
        # Build cover count once
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        block_ts_indices = []
        for b in best_solution:
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
        
        changed = False
        # Try removing redundant blocks (all their t-subsets covered by >= 2 blocks)
        for i in range(len(block_ts_indices)):
            if time.time() > deadline - 2.0:
                break
            ts_arr = block_ts_indices[i]
            if np.all(cover_count[ts_arr] >= 2):
                np.subtract.at(cover_count, ts_arr, 1)
                best_solution.pop(i)
                block_ts_indices.pop(i)
                if len(best_solution) < best_len:
                    best_len = len(best_solution)
                changed = True
                break
        
        if not changed:
            # Try 1-for-1 swaps: replace a block with one that covers its unique t-subsets
            # and has more overlap with other blocks
            for i in range(len(block_ts_indices)):
                if time.time() > deadline - 2.0:
                    break
                ts_arr_i = block_ts_indices[i]
                unique_mask = cover_count[ts_arr_i] == 1
                unique_ts = ts_arr_i[unique_mask]
                
                if len(unique_ts) == 0:
                    continue
                
                # Find candidates that cover all unique t-subsets
                candidates = set(ts_to_blocks[int(unique_ts[0])])
                for ts_idx in unique_ts[1:]:
                    candidates.intersection_update(ts_to_blocks[int(ts_idx)])
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                
                # Exclude current block
                cur_idx = get_block_idx(best_solution[i])
                candidates.discard(cur_idx)
                
                if not candidates:
                    continue
                
                # Score candidates by how many of their t-subsets are already covered
                # (higher overlap = better swap)
                best_cand = None
                best_score = -1
                cand_list = list(candidates)
                if len(cand_list) > 20:
                    random.shuffle(cand_list)
                    cand_list = cand_list[:20]
                
                for c in cand_list:
                    new_ts = block_to_ts_np[c]
                    # After swap: remove i, add c
                    # Check validity: all t-subsets still covered
                    # Unique t-subsets of i must be covered by c (guaranteed by construction)
                    # T-subsets of c that are NOT in i must be covered by others
                    # i.e., for ts in new_ts not in ts_arr_i, cover_count[ts] >= 1
                    # But we need to be careful: cover_count includes block i
                    # So for ts in new_ts \ ts_arr_i, we need cover_count[ts] >= 1
                    # (since removing i doesn't affect those)
                    # For ts in new_ts ∩ ts_arr_i, we need cover_count[ts] >= 2 
                    #   (one from i, one from c, but after removing i we still have c)
                    # Actually simpler: after removing i and adding c,
                    # for each ts: new_cover = cover_count[ts] - 1(if in ts_arr_i) + 1(if in new_ts)
                    # We need new_cover >= 1 for all ts
                    
                    # Quick check: min of (cover_count - indicator_in_i + indicator_in_c) >= 1
                    # For ts in ts_arr_i but not in new_ts: need cover_count[ts] >= 2
                    # For ts in new_ts but not in ts_arr_i: need cover_count[ts] >= 1
                    # For ts in both: need cover_count[ts] >= 1 (stays same)
                    # For ts in neither: need cover_count[ts] >= 1
                    
                    in_i = np.zeros(num_t_subsets, dtype=bool)
                    in_i[ts_arr_i] = True
                    in_c = np.zeros(num_t_subsets, dtype=bool)
                    in_c[new_ts] = True
                    
                    # Only check where coverage might drop
                    # ts in i but not c: need cover_count >= 2
                    only_i = in_i & ~in_c
                    if np.any(only_i) and np.min(cover_count[only_i]) < 2:
                        continue
                    
                    # ts in c but not i: need cover_count >= 1
                    only_c = in_c & ~in_i
                    if np.any(only_c) and np.min(cover_count[only_c]) < 1:
                        continue
                    
                    # Score: prefer blocks with more already-covered t-subsets (less new exposure)
                    score = int(np.sum(cover_count[new_ts] >= 2))
                    if score > best_score:
                        best_score = score
                        best_cand = c
                
                if best_cand is not None:
                    # Perform swap
                    np.subtract.at(cover_count, ts_arr_i, 1)
                    new_ts_arr = block_to_ts_np[best_cand]
                    np.add.at(cover_count, new_ts_arr, 1)
                    best_solution[i] = list(all_k_blocks[best_cand])
                    block_ts_indices[i] = new_ts_arr
                    if len(best_solution) < best_len:
                        best_len = len(best_solution)
                    changed = True
                    break
        
        if not changed:
            break

    return best_solution
