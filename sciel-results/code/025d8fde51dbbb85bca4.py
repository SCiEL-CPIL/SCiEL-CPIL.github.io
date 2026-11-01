import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit (2s buffer)

    # Helper function for combinations count
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

    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Build incidence structures
    # block_to_ts[j]: list of indices of t-subsets contained in block j
    # ts_to_blocks[i]: list of indices of blocks containing t-subset i
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)
        
    # Precompute a map from tuple(block) to its index in all_k_blocks for fast lookup
    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}
    
    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)

    # State for greedy
    gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
    is_uncovered = np.full(num_t_subsets, True, dtype=bool)
    is_selected = np.full(num_k_blocks, False, dtype=bool)
    
    solution_blocks = []
    
    # Greedy Phase (Deterministic)
    while True:
        if time.time() > deadline - 10.0:
            break
            
        max_idx = np.argmax(gain)
        if gain[max_idx] <= 0:
            break
            
        solution_blocks.append(list(all_k_blocks[max_idx]))
        is_selected[max_idx] = True
        gain[max_idx] = -1
        
        for ts_idx in block_to_ts[max_idx]:
            if is_uncovered[ts_idx]:
                is_uncovered[ts_idx] = False
                for b_idx in ts_to_blocks[ts_idx]:
                    if not is_selected[b_idx]:
                        gain[b_idx] -= 1
                    
    best_solution = solution_blocks
    best_len = len(solution_blocks)
    
    # Function to run a randomized greedy
    def randomized_greedy(seed):
        random.seed(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []
        
        while True:
            if time.time() > deadline - 5.0:
                return None
            
            max_gain = np.max(gain)
            if max_gain <= 0:
                break
                
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
                
            b_idx = candidates[random.randint(0, len(candidates) - 1)]
            
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

    # Run randomized greedy multiple times to find a better starting point
    num_runs = 50 # Reduced to leave time for ILS
    for i in range(num_runs):
        if time.time() > deadline - 10.0:
            break
            
        sol = randomized_greedy(i)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol
            
    # Optimized Local Search: Remove redundant blocks
    def optimize_redundancy(blocks):
        if not blocks:
            return blocks
            
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        
        # Map solution blocks to indices for speed
        sol_indices = []
        for b in blocks:
            idx = get_block_idx(b)
            sol_indices.append(idx)
            
        for idx in sol_indices:
            if idx >= 0:
                for ts_idx in block_to_ts[idx]:
                    cover_count[ts_idx] += 1
                    
        changed = True
        while changed:
            if time.time() > deadline - 2.0:
                return blocks
            changed = False
            to_remove = []
            for i, idx in enumerate(sol_indices):
                if idx < 0:
                    continue
                redundant = True
                for ts_idx in block_to_ts[idx]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(i)
            
            if to_remove:
                # Remove in reverse order to maintain indices
                for i in reversed(to_remove):
                    idx = sol_indices[i]
                    blocks.pop(i)
                    sol_indices.pop(i)
                    for ts_idx in block_to_ts[idx]:
                        cover_count[ts_idx] -= 1
                changed = True
        return blocks

    best_solution = optimize_redundancy(best_solution)
    
    # Optimized Local Search: 1-move swap
    def local_search_swap(blocks):
        if not blocks:
            return blocks, False
            
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        sol_indices = []
        for b in blocks:
            idx = get_block_idx(b)
            sol_indices.append(idx)
            if idx >= 0:
                for ts_idx in block_to_ts[idx]:
                    cover_count[ts_idx] += 1
                    
        sol_set = set(sol_indices)
        improved = False
        
        # Iterate multiple times until no improvement
        max_passes = 10
        for _ in range(max_passes):
            if time.time() > deadline - 2.0:
                return blocks, improved
            pass_improved = False
            
            for i in range(len(blocks)):
                idx = sol_indices[i]
                if idx < 0:
                    continue
                    
                # Find t-subsets uniquely covered by blocks[i]
                unique_ts = []
                for ts_idx in block_to_ts[idx]:
                    if cover_count[ts_idx] == 1:
                        unique_ts.append(ts_idx)
                        
                if not unique_ts:
                    continue
                
                # Sort unique_ts by frequency (smallest list first) to prune candidates faster
                unique_ts.sort(key=lambda x: len(ts_to_blocks[x]))
                
                # Find candidate blocks that cover ALL unique_ts
                # Start with the rarest one
                candidates = set(ts_to_blocks[unique_ts[0]])
                for ts_idx in unique_ts[1:]:
                    # Intersect with next rarest
                    # Optimization: if current candidates is smaller, iterate it
                    if len(candidates) < len(ts_to_blocks[ts_idx]):
                        candidates = candidates.intersection(ts_to_blocks[ts_idx])
                    else:
                        candidates = set(ts_to_blocks[ts_idx]).intersection(candidates)
                    if not candidates:
                        break
                        
                if not candidates:
                    continue
                    
                # Remove current block from candidates
                candidates.discard(idx)
                
                # Filter out blocks already in solution
                valid_candidates = [c for c in candidates if c not in sol_set]
                
                if not valid_candidates:
                    continue
                    
                # Pick the best candidate (max gain? or random?)
                # For speed, pick random or first. Let's pick one that maximizes new coverage?
                # Actually, any valid swap maintains validity. 
                # To improve future swaps, maybe pick one that covers more *other* unique ts?
                # Let's just pick randomly to avoid bias, or the one with highest current gain?
                # Since we are swapping, the "gain" concept is tricky. 
                # Let's just pick the first valid candidate.
                new_block_idx = valid_candidates[0]
                
                # Perform swap
                blocks[i] = list(all_k_blocks[new_block_idx])
                sol_indices[i] = new_block_idx
                
                # Update cover counts
                for ts_idx in block_to_ts[idx]:
                    cover_count[ts_idx] -= 1
                for ts_idx in block_to_ts[new_block_idx]:
                    cover_count[ts_idx] += 1
                    
                sol_set.discard(idx)
                sol_set.add(new_block_idx)
                
                pass_improved = True
                improved = True
                # Restart loop after a swap? Or continue?
                # Continuing is faster, but might miss better swaps.
                # Let's break and restart the pass to recalculate unique_ts for others?
                # No, unique_ts for others might change.
                # For efficiency, let's just break and restart the outer pass loop.
                break
                
            if not pass_improved:
                break
                
        return blocks, improved

    # Greedy Repair: Add blocks to cover uncovered t-subsets
    def greedy_repair(blocks, uncovered_ts_indices):
        if not uncovered_ts_indices:
            return blocks, True
            
        sol_set = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_set.add(idx)
                
        # We need to cover the specific uncovered_ts_indices
        # Greedy: pick block covering most uncovered ts
        # Create a list of uncovered ts
        unc_list = list(uncovered_ts_indices)
        
        # Map ts_idx to count of how many times it appears in unc_list? 
        # No, just iterate.
        
        # To speed up, we can filter blocks that cover at least one uncovered ts
        # And score them by how many uncovered ts they cover.
        
        # Since unc_list can be large, and blocks many, this is O(NumBlocks * K)
        # But we can stop if we cover all.
        
        current_unc_set = set(unc_list)
        
        while current_unc_set:
            if time.time() > deadline - 1.0:
                return blocks, False
                
            best_block_idx = -1
            best_cover_count = 0
            
            # Iterate all blocks? Too slow.
            # Better: Pick a random uncovered ts, look at blocks covering it, score them.
            # This is "Greedy with Randomization"
            
            # Pick a random uncovered ts
            target_ts = random.choice(list(current_unc_set))
            
            # Get all blocks covering target_ts
            candidates = ts_to_blocks[target_ts]
            
            # Score them
            for b_idx in candidates:
                if b_idx in sol_set:
                    continue
                    
                # Count how many uncovered ts this block covers
                cnt = 0
                for ts_idx in block_to_ts[b_idx]:
                    if ts_idx in current_unc_set:
                        cnt += 1
                        
                if cnt > best_cover_count:
                    best_cover_count = cnt
                    best_block_idx = b_idx
                    
            if best_block_idx == -1:
                # No block covers the target? Should not happen if valid.
                # Or all blocks are in sol_set.
                # If all blocks covering target_ts are in sol_set, then target_ts is already covered?
                # Contradiction. Unless we have a bug.
                # Break to avoid infinite loop.
                return blocks, False
                
            # Add best block
            blocks.append(list(all_k_blocks[best_block_idx]))
            sol_set.add(best_block_idx)
            
            # Remove covered ts from current_unc_set
            for ts_idx in block_to_ts[best_block_idx]:
                if ts_idx in current_unc_set:
                    current_unc_set.remove(ts_idx)
                    
        return blocks, True

    # Iterated Local Search Loop
    # We will run ILS for the remaining time.
    
    # Initial Local Search on best_solution
    best_solution, _ = local_search_swap(best_solution)
    best_solution = optimize_redundancy(best_solution)
    best_len = len(best_solution)
    
    # ILS
    while time.time() < deadline - 5.0:
        # 1. Perturb: Remove M random blocks
        M = random.randint(3, 8) # Remove 3-8 blocks
        if len(best_solution) <= M:
            continue
            
        perturbed = best_solution[:]
        # Remove M random blocks
        indices_to_remove = random.sample(range(len(perturbed)), M)
        for i in reversed(sorted(indices_to_remove)):
            perturbed.pop(i)
            
        # 2. Compute uncovered t-subsets
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in perturbed:
            idx = get_block_idx(b)
            if idx >= 0:
                for ts_idx in block_to_ts[idx]:
                    cover_count[ts_idx] += 1
                    
        uncovered = np.where(cover_count == 0)[0]
        
        # 3. Repair
        repaired, success = greedy_repair(perturbed, uncovered)
        if not success:
            continue
            
        # 4. Local Search
        repaired, _ = local_search_swap(repaired)
        repaired = optimize_redundancy(repaired)
        
        # 5. Update Best
        if len(repaired) < best_len:
            best_len = len(repaired)
            best_solution = repaired
            
    # Final cleanup
    best_solution = optimize_redundancy(best_solution)
    
    return best_solution
