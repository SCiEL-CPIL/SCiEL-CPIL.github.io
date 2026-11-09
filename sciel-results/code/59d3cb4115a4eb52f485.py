import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    DEADLINE = start_time + 115.0  # Safety margin for 120s limit
    
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    all_t = list(itertools.combinations(points, t))
    N_t = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}
    
    # Precompute block to t-subset mappings
    # block_tsubs[i] is a list of t-subset indices covered by block i
    block_tsubs_list = []
    for b in all_blocks:
        subs = [t_idx[s] for s in itertools.combinations(b, t)]
        block_tsubs_list.append(subs)
        
    # tsub_blocks[j] is a list of block indices covering t-subset j
    tsub_blocks_list = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_tsubs_list):
        for si in subs:
            tsub_blocks_list[si].append(i)
            
    # Convert to NumPy arrays for vectorized operations
    # We store them as arrays of arrays. 
    # Note: Uneven lengths are handled by object arrays or lists of arrays.
    # Lists of arrays are faster for indexing specific blocks than object arrays in some cases.
    block_tsubs_np = [np.array(lst, dtype=np.int32) for lst in block_tsubs_list]
    tsub_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_blocks_list]
    
    # For the greedy phase, we need fast access to the list of blocks covering a t-subset
    # tsub_blocks_np[si] is the array of block indices.

    def run_greedy_indices(randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)
            
        # block_gain[i] = number of uncovered t-subsets covered by block i
        # Initialize with full count since nothing is covered
        block_gain = np.array([len(subs) for subs in block_tsubs_list], dtype=np.int32)
        is_covered = np.zeros(N_t, dtype=np.int8)
        uncovered_count = N_t
        selected_indices = []
        
        while uncovered_count > 0 and time.time() < DEADLINE:
            max_gain = np.max(block_gain)
            if max_gain <= 0:
                break
            
            if randomize:
                candidates = np.where(block_gain == max_gain)[0]
                best_block_idx = random.choice(candidates.tolist())
            else:
                best_block_idx = int(np.argmax(block_gain))
                
            selected_indices.append(best_block_idx)
            
            # Get the t-subsets covered by this block
            subs_np = block_tsubs_np[best_block_idx]
            
            # Find which of these are newly covered
            # is_covered[subs_np] gives the status of these t-subsets
            newly_covered_mask = is_covered[subs_np] == 0
            newly_covered_indices = subs_np[newly_covered_mask]
            
            if len(newly_covered_indices) > 0:
                is_covered[newly_covered_indices] = 1
                uncovered_count -= len(newly_covered_indices)
                
                # Update gains: For each newly covered t-subset, decrement gain of all blocks covering it
                # We need to collect all block indices that need decrementing
                # This can be a large list, so we build it efficiently
                all_block_indices_to_decrement = []
                for si in newly_covered_indices:
                    # tsub_blocks_np[si] is an array of block indices
                    all_block_indices_to_decrement.extend(tsub_blocks_np[si].tolist())
                
                if all_block_indices_to_decrement:
                    # Use np.subtract.at for vectorized decrement
                    # This is much faster than a Python loop
                    indices_np = np.array(all_block_indices_to_decrement, dtype=np.int32)
                    np.subtract.at(block_gain, indices_np, 1)
                    
        return selected_indices

    def anneal_solution(initial_indices, time_limit=5.0):
        """
        Simulated Annealing with vectorized delta calculations.
        """
        if not initial_indices:
            return initial_indices
            
        M = 1000 # Penalty for uncovered t-subsets
        T = 2.0
        alpha = 0.97
        T_min = 1e-3
        
        # State
        selected = list(initial_indices)
        selected_set = set(selected)
        cov = np.zeros(N_t, dtype=np.int32)
        
        # Initialize coverage
        for b_idx in selected:
            cov[block_tsubs_np[b_idx]] += 1
            
        uncovered_count = int(np.sum(cov == 0))
        current_cost = len(selected) + M * uncovered_count
        
        best_cost = current_cost
        best_selected = list(selected)
        
        start = time.time()
        
        while T > T_min and (time.time() - start) < time_limit and time.time() < DEADLINE:
            # Generate Move: 35% Remove, 25% Add, 40% Swap
            move_type = random.random()
            
            if move_type < 0.35 and selected:
                # Remove a block
                b_idx = random.choice(selected)
                subs_np = block_tsubs_np[b_idx]
                
                # Vectorized delta computation
                # How many t-subsets become uncovered? Those with cov == 1
                delta_uncovered = int(np.sum(cov[subs_np] == 1))
                delta_cost = -1 + M * delta_uncovered
                
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    # Accept
                    selected.remove(b_idx)
                    selected_set.discard(b_idx)
                    cov[subs_np] -= 1
                    uncovered_count += delta_uncovered
                    current_cost += delta_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_selected = list(selected)
                        
            elif move_type < 0.60:
                # Add a block
                if uncovered_count > 0:
                    # Pick a random uncovered t-subset
                    # Using np.where is O(N_t), which is ~11k, acceptable.
                    # Alternative: maintain a list of uncovered indices for O(1) pick, but update is costly.
                    uncovered_mask = (cov == 0).nonzero()[0]
                    if len(uncovered_mask) > 0:
                        target_si = int(uncovered_mask[random.randint(0, len(uncovered_mask)-1)])
                        candidate_blocks = tsub_blocks_np[target_si]
                        if len(candidate_blocks) > 0:
                            b_idx = int(candidate_blocks[random.randint(0, len(candidate_blocks)-1)])
                            if b_idx not in selected_set:
                                subs_np = block_tsubs_np[b_idx]
                                # How many newly covered? Those with cov == 0
                                newly_covered = int(np.sum(cov[subs_np] == 0))
                                delta_cost = 1 - M * newly_covered
                                
                                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                                    # Accept
                                    selected.append(b_idx)
                                    selected_set.add(b_idx)
                                    cov[subs_np] += 1
                                    uncovered_count -= newly_covered
                                    current_cost += delta_cost
                                    
                                    if current_cost < best_cost:
                                        best_cost = current_cost
                                        best_selected = list(selected)
                else:
                    # No uncovered, add random block (likely rejected)
                    b_idx = random.randint(0, N_blocks - 1)
                    if b_idx not in selected_set:
                        delta_cost = 1
                        if random.random() < np.exp(-delta_cost / T):
                            selected.append(b_idx)
                            selected_set.add(b_idx)
                            cov[block_tsubs_np[b_idx]] += 1
                            current_cost += delta_cost
                            
            else:
                # Swap: Remove b1, Add b2
                if len(selected) >= 2:
                    b1_idx = random.choice(selected)
                    b1_subs = block_tsubs_np[b1_idx]
                    
                    # Calculate delta for removing b1
                    # Uncovered increase: t-subsets in b1 with cov == 1
                    newly_uncovered_from_b1 = int(np.sum(cov[b1_subs] == 1))
                    
                    # Pick b2
                    # Pick a random t-subset from b1, then a random block covering it
                    if len(b1_subs) > 0:
                        target_si = int(b1_subs[random.randint(0, len(b1_subs)-1)])
                        candidate_blocks = tsub_blocks_np[target_si]
                        if len(candidate_blocks) > 0:
                            b2_idx = int(candidate_blocks[random.randint(0, len(candidate_blocks)-1)])
                            
                            if b2_idx not in selected_set:
                                b2_subs = block_tsubs_np[b2_idx]
                                
                                # Calculate delta for adding b2
                                # Newly covered: t-subsets in b2 with cov == 0 (before removing b1? No, after removing b1)
                                # We need to simulate:
                                # 1. Remove b1: cov' = cov - 1 (for b1_subs)
                                # 2. Add b2: cov'' = cov' + 1 (for b2_subs)
                                
                                # Uncovered count change:
                                # Start: U
                                # After remove b1: U + newly_uncovered_from_b1
                                # After add b2: 
                                #   Some of those newly uncovered might be covered by b2.
                                #   Some previously covered (cov>0) might remain covered.
                                #   Some previously uncovered (cov==0) might be covered by b2.
                                
                                # Let's compute directly on the simulated coverage for the affected sets.
                                # Affected t-subsets: union(b1_subs, b2_subs)
                                # This set is small (max 56+56).
                                
                                # Create a mask of affected indices? 
                                # Simpler: iterate through the small list of unique t-subsets in b1 and b2.
                                # But we want vectorized.
                                
                                # Let's use a temporary array for the affected indices.
                                # union_set = np.unique(np.concatenate([b1_subs, b2_subs]))
                                # This is fast enough for ~100 elements.
                                union_subs = np.unique(np.concatenate([b1_subs, b2_subs]))
                                
                                # Current cov values
                                cov_before = cov[union_subs]
                                
                                # Simulate removal of b1
                                # b1_subs is in union_subs. We need to decrement cov for those.
                                # We can create a mask for b1_subs in union_subs
                                # mask_b1 = np.isin(union_subs, b1_subs)
                                # This is O(N*M). 
                                # Alternative: use a dictionary or set for b1_subs for fast lookup? 
                                # Since size is small, np.isin is fine.
                                
                                mask_b1 = np.isin(union_subs, b1_subs)
                                mask_b2 = np.isin(union_subs, b2_subs)
                                
                                cov_after_remove = cov_before.copy()
                                cov_after_remove[mask_b1] -= 1
                                
                                # Count newly uncovered (cov_after_remove == 0)
                                newly_uncovered_actual = int(np.sum(cov_after_remove == 0))
                                
                                # Simulate addition of b2
                                cov_after_add = cov_after_remove.copy()
                                cov_after_add[mask_b2] += 1
                                
                                # Count uncovered after add
                                uncovered_after = int(np.sum(cov_after_add == 0))
                                
                                # Delta uncovered = uncovered_after - (uncovered_count - newly_uncovered_from_b1) ?
                                # No, let's just compute the net change in uncovered count for these specific indices.
                                # Original uncovered in union: sum(cov_before == 0)
                                # New uncovered in union: sum(cov_after_add == 0)
                                delta_uncovered = uncovered_after - int(np.sum(cov_before == 0))
                                
                                delta_cost = -1 + 1 + M * delta_uncovered # Net block change is 0
                                
                                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                                    # Accept
                                    # Remove b1
                                    selected.remove(b1_idx)
                                    selected_set.discard(b1_idx)
                                    cov[b1_subs] -= 1
                                    
                                    # Add b2
                                    selected.append(b2_idx)
                                    selected_set.add(b2_idx)
                                    cov[b2_subs] += 1
                                    
                                    uncovered_count += delta_uncovered
                                    current_cost += delta_cost
                                    
                                    if current_cost < best_cost:
                                        best_cost = current_cost
                                        best_selected = list(selected)
            
            T *= alpha
            
        # Repair: Ensure validity
        # Greedily add blocks to cover any remaining uncovered t-subsets
        uncovered_tsubs = [i for i in range(N_t) if cov[i] == 0]
        while uncovered_tsubs:
            # Find block covering most uncovered
            best_b = -1
            max_cov = 0
            # Sample some blocks if list is too long?
            # For now, check blocks covering the first few uncovered t-subsets
            candidate_blocks = set()
            for si in uncovered_tsubs[:10]:
                candidate_blocks.update(tsub_blocks_np[si].tolist())
            
            for b_idx in candidate_blocks:
                if b_idx in selected_set:
                    continue
                subs_np = block_tsubs_np[b_idx]
                count = int(np.sum(cov[subs_np] == 0))
                if count > max_cov:
                    max_cov = count
                    best_b = b_idx
            
            if best_b == -1:
                break # Should not happen if problem is solvable
                
            selected.append(best_b)
            selected_set.add(best_b)
            subs_np = block_tsubs_np[best_b]
            # Update cov
            newly_covered_mask = cov[subs_np] == 0
            cov[subs_np] += 1
            uncovered_count -= int(np.sum(newly_covered_mask))
            
            # Re-calculate uncovered list
            uncovered_tsubs = [i for i in range(N_t) if cov[i] == 0]
            
        # Final Local Search: Reduction
        improved = True
        while improved:
            improved = False
            for b_idx in list(selected):
                subs_np = block_tsubs_np[b_idx]
                # Can remove if no t-subset has cov == 1
                if int(np.sum(cov[subs_np] == 1)) == 0:
                    selected.remove(b_idx)
                    selected_set.discard(b_idx)
                    cov[subs_np] -= 1
                    improved = True
                    break # Restart loop
                    
        return selected

    # Main Loop
    best_sol_idx = None
    best_len = float('inf')
    
    # 1. Deterministic Greedy
    sol_idx = run_greedy_indices(randomize=False)
    sol_idx = anneal_solution(sol_idx, time_limit=10.0)
    best_sol_idx = sol_idx
    best_len = len(sol_idx)
    
    # 2. Randomized Restarts
    restart_count = 0
    while time.time() < DEADLINE - 15:
        seed = random.randint(0, 2**31)
        cand_idx = run_greedy_indices(randomize=True, seed=seed)
        # Limit time per restart
        remaining = DEADLINE - time.time()
        if remaining < 5:
            break
        time_limit = min(5.0, remaining - 10.0) # Leave 10s for repair/final checks
        if time_limit < 0.5:
            break
            
        cand_idx = anneal_solution(cand_idx, time_limit=time_limit)
        
        if len(cand_idx) < best_len:
            best_sol_idx = cand_idx
            best_len = len(cand_idx)
            
        restart_count += 1

    return [all_blocks[i] for i in best_sol_idx]
