import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
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
    
    # Build incidence structures
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)
    
    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}
    
    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)
    
    def greedy_construction(seed=None, weight_temp=0.0):
        """
        Weighted greedy: pick block maximizing sum of weights for uncovered t-subsets.
        If weight_temp > 0, add random perturbation for diversity.
        """
        if seed is not None:
            random.seed(seed)
        
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []
        
        while True:
            if time.time() > deadline - 8.0:
                return None
            
            max_idx = np.argmax(gain)
            if gain[max_idx] <= 0:
                break
            
            # Add random perturbation for diversity
            if weight_temp > 0:
                # Perturb top candidates
                top_k = 20
                top_indices = np.argsort(gain)[-top_k:]
                gains = gain[top_indices]
                max_g = np.max(gains)
                # Sample from top candidates with probability proportional to exp((gain - max_g)/temp)
                log_probs = (gains - max_g) / weight_temp
                log_probs -= np.max(log_probs)  # numerical stability
                probs = np.exp(log_probs)
                probs /= np.sum(probs)
                chosen_local = np.random.choice(len(top_indices), p=probs)
                b_idx = top_indices[chosen_local]
            else:
                b_idx = max_idx
            
            if is_selected[b_idx]:
                # Shouldn't happen if we mark selected, but be safe
                continue
            
            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1
            
            for ts_idx in block_to_ts[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    for nb_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[nb_idx]:
                            gain[nb_idx] -= 1
                            
        return sol
    
    def remove_redundant(blocks):
        """Remove blocks that are entirely redundant (all their t-subsets covered by others)."""
        if not blocks:
            return blocks
        
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts_index_map[ts]] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 2.0:
                    return blocks
                
                # Check if block i is redundant
                redundant = True
                for ts in itertools.combinations(blocks[i], t):
                    if cover_count[ts_index_map[ts]] == 1:
                        redundant = False
                        break
                        
                if redundant:
                    block_to_remove = blocks[i]
                    blocks.pop(i)
                    for ts in itertools.combinations(block_to_remove, t):
                        cover_count[ts_index_map[ts]] -= 1
                    changed = True
                    break
                    
        return blocks
    
    def local_search(blocks, max_passes=5):
        """
        Local search: try to swap a block with another that covers its unique t-subsets.
        Also try to remove redundant blocks.
        """
        if not blocks:
            return blocks
        
        # Build cover counts
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts_index_map[ts]] += 1
        
        # Track which blocks are in the solution
        sol_block_set = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_set.add(idx)
        
        improved = True
        passes = 0
        
        while improved and passes < max_passes:
            if time.time() > deadline - 3.0:
                break
            improved = False
            passes += 1
            
            # First pass: try removals
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline - 3.0:
                    break
                
                # Check if block i is redundant
                redundant = True
                for ts in itertools.combinations(blocks[i], t):
                    if cover_count[ts_index_map[ts]] == 1:
                        redundant = False
                        break
                        
                if redundant:
                    block_to_remove = blocks[i]
                    blocks.pop(i)
                    for ts in itertools.combinations(block_to_remove, t):
                        cover_count[ts_index_map[ts]] -= 1
                    improved = True
                    # Restart the loop since indices changed
                    break
            else:
                # No removals, try swaps
                for i in range(len(blocks)):
                    if time.time() > deadline - 3.0:
                        break
                    
                    # Find unique t-subsets covered only by blocks[i]
                    unique_ts = []
                    for ts in itertools.combinations(blocks[i], t):
                        idx = ts_index_map[ts]
                        if cover_count[idx] == 1:
                            unique_ts.append(idx)
                            
                    if not unique_ts:
                        # Block is fully redundant, should have been removed
                        # But let's be safe and mark for removal
                        block_to_remove = blocks[i]
                        blocks.pop(i)
                        for ts in itertools.combinations(block_to_remove, t):
                            cover_count[ts_index_map[ts]] -= 1
                        improved = True
                        break
                    
                    # Find candidate blocks that cover all unique_ts
                    # Start with blocks covering the first unique ts
                    candidates = set(ts_to_blocks[unique_ts[0]])
                    for ts_idx in unique_ts[1:]:
                        candidates.intersection_update(ts_to_blocks[ts_idx])
                        if not candidates:
                            break
                    
                    if not candidates:
                        continue
                    
                    # Remove current block from candidates
                    current_idx = get_block_idx(blocks[i])
                    candidates.discard(current_idx)
                    
                    # Filter out blocks already in solution
                    valid_candidates = [c for c in candidates if c not in sol_block_set]
                    
                    if not valid_candidates:
                        continue
                    
                    # Try each candidate (or a subset for speed)
                    random.shuffle(valid_candidates)
                    for new_idx in valid_candidates[:10]:  # Try up to 10 candidates
                        # Check if this swap is valid
                        # Temporarily remove old block and add new block
                        old_block = blocks[i]
                        new_block = list(all_k_blocks[new_idx])
                        
                        # Check if new_block's t-subsets are all covered after removal
                        # We need to check: after removing old_block, is every t-subset of new_block still covered?
                        # Actually, we need to check that the union of (all blocks except i) ∪ {new_block} covers everything.
                        # Since we're only replacing blocks[i], we need to check that any t-subset that was uniquely covered by blocks[i] is now covered by new_block.
                        # And any t-subset that was covered by both blocks[i] and others remains covered.
                        # The critical check: for each t-subset, cover_count after swap >= 1.
                        
                        # Efficient check: 
                        # 1. Subtract old_block's contributions
                        # 2. Check if any count becomes 0
                        # 3. Add new_block's contributions
                        # 4. If valid, commit; else revert
                        
                        # But this is O(C(k,t)) per candidate. Let's do it.
                        
                        # Save current state
                        old_counts = cover_count.copy()
                        
                        for ts in itertools.combinations(old_block, t):
                            cover_count[ts_index_map[ts]] -= 1
                        
                        # Check if any count is 0
                        if np.min(cover_count) < 0:
                            cover_count[:] = old_counts
                            continue
                        
                        # Check if any count is 0
                        if np.any(cover_count == 0):
                            # This swap would uncover some t-subset
                            cover_count[:] = old_counts
                            continue
                        
                        # Valid swap
                        for ts in itertools.combinations(new_block, t):
                            cover_count[ts_index_map[ts]] += 1
                        
                        blocks[i] = new_block
                        sol_block_set.remove(current_idx)
                        sol_block_set.add(new_idx)
                        improved = True
                        break  # Break inner loop, continue outer loop
                        
            if improved:
                # Rebuild cover counts and sol_block_set after changes
                cover_count = np.zeros(num_t_subsets, dtype=np.int32)
                for b in blocks:
                    for ts in itertools.combinations(b, t):
                        cover_count[ts_index_map[ts]] += 1
                sol_block_set = set()
                for b in blocks:
                    idx = get_block_idx(b)
                    if idx >= 0:
                        sol_block_set.add(idx)
        
        return blocks
    
    # Main algorithm
    best_solution = None
    best_len = float('inf')
    
    # Phase 1: Run multiple greedy constructions with different random seeds
    num_greedy_runs = 50
    for i in range(num_greedy_runs):
        if time.time() > deadline - 10.0:
            break
        
        sol = greedy_construction(seed=i, weight_temp=0.0)
        if sol is not None:
            sol = remove_redundant(sol)
            if len(sol) < best_len:
                best_len = len(sol)
                best_solution = sol
                # Try local search on this solution
                if time.time() < deadline - 5.0:
                    improved_sol = local_search(sol, max_passes=3)
                    if len(improved_sol) < best_len:
                        best_len = len(improved_sol)
                        best_solution = improved_sol
    
    # Phase 2: Try some randomized greedy with perturbation
    for i in range(30):
        if time.time() > deadline - 10.0:
            break
        
        # Use different temperature values
        temp = 1.0 + i * 0.5
        sol = greedy_construction(seed=1000 + i, weight_temp=temp)
        if sol is not None:
            sol = remove_redundant(sol)
            if len(sol) < best_len:
                best_len = len(sol)
                best_solution = sol
                if time.time() < deadline - 5.0:
                    improved_sol = local_search(sol, max_passes=3)
                    if len(improved_sol) < best_len:
                        best_len = len(improved_sol)
                        best_solution = improved_sol
    
    # Phase 3: Local search on the best solution
    if best_solution is not None and time.time() < deadline - 5.0:
        improved_sol = local_search(best_solution, max_passes=10)
        if len(improved_sol) < best_len:
            best_len = len(improved_sol)
            best_solution = improved_sol
    
    # Phase 4: Final redundancy removal
    if best_solution is not None:
        best_solution = remove_redundant(best_solution)
    
    # Fallback: if no solution found, return a simple greedy solution
    if best_solution is None:
        sol = greedy_construction(seed=0, weight_temp=0.0)
        best_solution = sol if sol is not None else []
    
    return best_solution
