import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for safety

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

    # Map each block to the indices of t-subsets it covers
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for j, block in enumerate(all_k_blocks):
        ts_indices = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_indices.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_indices)

    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}

    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)

    def check_time():
        return time.time() < deadline

    def greedy_complete(initial_blocks, seed=0):
        """
        Complete the covering starting from initial_blocks.
        Uses greedy set cover on the remaining uncovered t-subsets.
        """
        random.seed(seed)
        
        # Initialize coverage state
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        
        # Mark initial blocks
        sol_indices = []
        for b in initial_blocks:
            idx = get_block_idx(b)
            if idx >= 0 and not is_selected[idx]:
                is_selected[idx] = True
                sol_indices.append(idx)
                for ts_idx in block_to_ts[idx]:
                    cover_count[ts_idx] += 1
        
        # Calculate initial gain: how many *new* t-subsets each block covers
        # gain[i] = number of ts in block i where cover_count[ts] == 0
        gain = np.zeros(num_k_blocks, dtype=np.int32)
        for i in range(num_k_blocks):
            if not is_selected[i]:
                g = 0
                for ts_idx in block_to_ts[i]:
                    if cover_count[ts_idx] == 0:
                        g += 1
                gain[i] = g

        sol_list = [list(all_k_blocks[idx]) for idx in sol_indices]
        
        # Greedy loop
        while check_time():
            # Find best block
            # Only consider unselected blocks with gain > 0
            valid_mask = (~is_selected) & (gain > 0)
            if not np.any(valid_mask):
                break
                
            max_gain = int(np.max(gain[valid_mask]))
            candidates = np.where(valid_mask & (gain == max_gain))[0]
            
            if len(candidates) == 0:
                break
                
            b_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            
            sol_list.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            sol_indices.append(b_idx)
            
            # Update coverage and gains
            newly_covered = []
            for ts_idx in block_to_ts[b_idx]:
                if cover_count[ts_idx] == 0:
                    cover_count[ts_idx] = 1
                    newly_covered.append(ts_idx)
                else:
                    cover_count[ts_idx] += 1
            
            if not newly_covered:
                continue
                
            # Update gain for all blocks that cover these newly covered ts
            # If a block covers a ts that just became covered, its gain decreases by 1
            for ts_idx in newly_covered:
                for nb_idx in ts_to_blocks[ts_idx]:
                    if not is_selected[nb_idx]:
                        gain[nb_idx] -= 1
                        
        return sol_list

    def optimize_redundancy(blocks):
        """Remove redundant blocks."""
        if not blocks:
            return []
        
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        block_indices = []
        
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                block_indices.append(idx)
        
        n_blocks = len(block_indices)
        if n_blocks == 0:
            return []

        for bidx in block_indices:
            for ts_idx in block_to_ts[bidx]:
                cover_count[ts_idx] += 1

        changed = True
        while changed and check_time():
            changed = False
            new_block_indices = []
            for bidx in block_indices:
                is_unique = False
                for ts_idx in block_to_ts[bidx]:
                    if cover_count[ts_idx] == 1:
                        is_unique = True
                        break
                if is_unique:
                    new_block_indices.append(bidx)
                else:
                    for ts_idx in block_to_ts[bidx]:
                        cover_count[ts_idx] -= 1
                    changed = True
            
            block_indices = new_block_indices
            if not block_indices:
                break

        return [list(all_k_blocks[idx]) for idx in block_indices]

    def local_search_1swap(blocks, max_rounds=5):
        """Fast 1-swap local search."""
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)
                for ts_idx in block_to_ts[idx]:
                    cover_count[ts_idx] += 1
        
        improved = False
        n = len(blocks)
        if n == 0:
            return blocks, False
            
        for _ in range(max_rounds):
            if not check_time():
                break
            round_improved = False
            order = list(range(n))
            random.shuffle(order)
            
            for i in order:
                if not check_time():
                    break
                old_b_idx = get_block_idx(blocks[i])
                if old_b_idx < 0:
                    continue
                
                unique_ts = [ts_idx for ts_idx in block_to_ts[old_b_idx] if cover_count[ts_idx] == 1]
                if not unique_ts:
                    continue
                
                candidates = set(ts_to_blocks[unique_ts[0]])
                for ts_idx in unique_ts[1:]:
                    candidates.intersection_update(ts_to_blocks[ts_idx])
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                
                valid_candidates = [c for c in candidates if c not in sol_block_indices]
                if not valid_candidates:
                    continue
                
                best_new = None
                best_gain = -1
                # Sample candidates if too many
                if len(valid_candidates) > 10:
                    valid_candidates = random.sample(valid_candidates, 10)
                
                for c in valid_candidates:
                    gain_val = 0
                    for ts_idx in block_to_ts[c]:
                        if cover_count[ts_idx] == 0:
                            gain_val += 1
                    if gain_val > best_gain:
                        best_gain = gain_val
                        best_new = c
                
                if best_new is None:
                    continue
                    
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

    # --- Phase 1: Construct Maximal t-Packing Scaffold ---
    # A packing is a set of blocks with disjoint t-subsets.
    # This provides a strong initial structure.
    packing = []
    used_ts = set()
    
    # Sort blocks by some heuristic? Random is fine for packing.
    block_order = list(range(num_k_blocks))
    random.seed(42)
    random.shuffle(block_order)
    
    for b_idx in block_order:
        if not check_time():
            break
        b = all_k_blocks[b_idx]
        b_ts = block_to_ts[b_idx]
        
        # Check if disjoint from used_ts
        # Since b_ts are indices, we can check intersection
        if not any(ts in used_ts for ts in b_ts):
            packing.append(list(b))
            for ts in b_ts:
                used_ts.add(ts)
                
        if len(packing) >= 50: # Limit packing size to keep completion fast
            break

    # --- Phase 2: Greedy Completion ---
    best_solution = greedy_complete(packing, seed=1)
    best_len = len(best_solution)
    best_solution = optimize_redundancy(best_solution)
    best_len = len(best_solution)

    # --- Phase 3: Local Search ---
    improved = True
    rounds = 0
    while improved and check_time() and rounds < 10:
        improved = False
        rounds += 1
        best_solution, imp1 = local_search_1swap(best_solution, max_rounds=3)
        if imp1:
            improved = True
            best_solution = optimize_redundancy(best_solution)
            if len(best_solution) < best_len:
                best_len = len(best_solution)

    # --- Phase 4: Iterated Local Search (ILS) ---
    # If we have time, try restarting with different packings or perturbations
    if check_time():
        for restart in range(5):
            if not check_time():
                break
            
            # New random packing
            random.seed(100 + restart)
            block_order = list(range(num_k_blocks))
            random.shuffle(block_order)
            
            packing_r = []
            used_ts_r = set()
            for b_idx in block_order:
                if not check_time():
                    break
                b_ts = block_to_ts[b_idx]
                if not any(ts in used_ts_r for ts in b_ts):
                    packing_r.append(list(all_k_blocks[b_idx]))
                    for ts in b_ts:
                        used_ts_r.add(ts)
                if len(packing_r) >= 30:
                    break

            sol_r = greedy_complete(packing_r, seed=100 + restart)
            sol_r = optimize_redundancy(sol_r)
            
            # Local search on new solution
            sol_r, _ = local_search_1swap(sol_r, max_rounds=3)
            sol_r = optimize_redundancy(sol_r)
            
            if len(sol_r) < best_len:
                best_len = len(sol_r)
                best_solution = sol_r

    return best_solution
