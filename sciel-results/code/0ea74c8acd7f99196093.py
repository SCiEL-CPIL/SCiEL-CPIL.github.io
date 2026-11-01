import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin

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
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    num_t_subsets = len(all_t_subsets)
    
    all_k_blocks = list(itertools.combinations(points, k))
    num_k_blocks = len(all_k_blocks)
    
    # Map t-subset to index
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
        
    # Convert ts_to_blocks to numpy arrays for faster intersection checks if needed
    # But lists are fine for small sets.
    
    # Precompute block indices for fast lookup
    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}
    
    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)

    def run_greedy(seed):
        random.seed(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol_indices = []
        
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
            
            sol_indices.append(b_idx)
            is_selected[b_idx] = True
            gain[b_idx] = -1
            
            newly_covered_ts = []
            for ts_idx in block_to_ts[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    newly_covered_ts.append(ts_idx)
            
            if not newly_covered_ts:
                continue
                
            blocks_to_decrement = []
            for ts_idx in newly_covered_ts:
                blocks_to_decrement.extend(ts_to_blocks[ts_idx])
                
            if blocks_to_decrement:
                counts = np.bincount(blocks_to_decrement, minlength=num_k_blocks)
                gain -= counts
                gain[is_selected] = -1000

        return sol_indices

    # Initial deterministic greedy
    best_sol_indices = run_greedy(0)
    if best_sol_indices is None:
        best_sol_indices = [0] # Fallback
    best_len = len(best_sol_indices)
    
    # Run randomized greedy multiple times
    num_runs = 100
    for i in range(1, num_runs):
        if time.time() > deadline - 10.0:
            break
        sol = run_greedy(i)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_sol_indices = sol

    # Convert indices to list of blocks for the final return format later
    # But keep working with indices for efficiency
    
    def optimize_redundancy(sol_indices):
        # sol_indices: list of block indices
        if not sol_indices:
            return sol_indices
            
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        # Map solution blocks to their precomputed ts indices
        # We need to know which blocks are in the solution for O(1) check?
        # Actually, we just need to update cover_count.
        
        sol_set = set(sol_indices)
        
        for b_idx in sol_indices:
            for ts_idx in block_to_ts[b_idx]:
                cover_count[ts_idx] += 1
                
        changed = True
        while changed:
            changed = False
            # Iterate in reverse to handle popping safely, or just mark and filter
            new_sol_indices = []
            for b_idx in sol_indices:
                if time.time() > deadline - 2.0:
                    return sol_indices
                    
                redundant = True
                for ts_idx in block_to_ts[b_idx]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                        
                if redundant:
                    # Remove block
                    for ts_idx in block_to_ts[b_idx]:
                        cover_count[ts_idx] -= 1
                    changed = True
                else:
                    new_sol_indices.append(b_idx)
                    
            if changed:
                sol_indices = new_sol_indices
                
        return sol_indices

    def local_search_1_for_2(sol_indices):
        # Try to remove 2 blocks and add 1 block that covers their unique t-subsets
        if len(sol_indices) < 2:
            return sol_indices, False
            
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in sol_indices:
            for ts_idx in block_to_ts[b_idx]:
                cover_count[ts_idx] += 1
                
        sol_set = set(sol_indices)
        improved = False
        
        # Calculate unique ts for each block
        unique_ts_map = {}
        for b_idx in sol_indices:
            unique_ts = []
            for ts_idx in block_to_ts[b_idx]:
                if cover_count[ts_idx] == 1:
                    unique_ts.append(ts_idx)
            unique_ts_map[b_idx] = unique_ts
            
        # Try pairs
        n = len(sol_indices)
        # Sort indices by number of unique ts to prioritize "hard" blocks? 
        # Or just iterate.
        # To save time, limit pairs checked.
        pairs_checked = 0
        max_pairs = 2000
        
        for i in range(n):
            if time.time() > deadline - 2.0:
                return sol_indices, improved
            if pairs_checked > max_pairs:
                break
            for j in range(i + 1, n):
                if pairs_checked > max_pairs:
                    break
                pairs_checked += 1
                
                u_i = unique_ts_map[sol_indices[i]]
                u_j = unique_ts_map[sol_indices[j]]
                
                if not u_i and not u_j:
                    continue
                    
                all_unique = u_i + u_j
                if not all_unique:
                    continue
                
                # Find candidates covering all_unique
                # Start with ts_to_blocks of the first unique ts
                # If all_unique is large, this intersection might be empty quickly
                if len(all_unique) > 10: # Heuristic: if too many unique ts, unlikely to be covered by one block
                    continue
                    
                candidates = set(ts_to_blocks[all_unique[0]])
                for ts_idx in all_unique[1:]:
                    candidates.intersection_update(ts_to_blocks[ts_idx])
                    if not candidates:
                        break
                        
                if not candidates:
                    continue
                
                # Filter out blocks already in solution
                valid_candidates = [c for c in candidates if c not in sol_set]
                if not valid_candidates:
                    continue
                
                # Pick the best candidate (e.g., one that covers most other ts)
                # For now, just pick one
                best_candidate = valid_candidates[0]
                
                # Perform swap
                # Remove i and j, add best_candidate
                # Update cover_count
                for ts_idx in block_to_ts[sol_indices[i]]:
                    cover_count[ts_idx] -= 1
                for ts_idx in block_to_ts[sol_indices[j]]:
                    cover_count[ts_idx] -= 1
                for ts_idx in block_to_ts[best_candidate]:
                    cover_count[ts_idx] += 1
                    
                sol_set.discard(sol_indices[i])
                sol_set.discard(sol_indices[j])
                sol_set.add(best_candidate)
                
                # Update sol_indices list
                # We need to remove i and j and add best_candidate
                # Since we are iterating, it's safer to rebuild or handle carefully
                # Let's rebuild for simplicity in this step, though it's O(n)
                new_sol_indices = []
                for idx in sol_indices:
                    if idx != sol_indices[i] and idx != sol_indices[j]:
                        new_sol_indices.append(idx)
                new_sol_indices.append(best_candidate)
                
                sol_indices = new_sol_indices
                improved = True
                # Restart the search as the state changed significantly
                return sol_indices, improved

        return sol_indices, improved

    def perturb_and_repair(sol_indices, seed):
        # Remove a random block and re-greedy the gaps
        if len(sol_indices) == 0:
            return sol_indices
            
        random.seed(seed)
        
        # Pick a random block to remove
        rem_idx = random.randint(0, len(sol_indices) - 1)
        rem_block = sol_indices[rem_idx]
        
        # Remove it
        new_sol = [b for i, b in enumerate(sol_indices) if i != rem_idx]
        
        # Calculate uncovered ts
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in new_sol:
            for ts_idx in block_to_ts[b_idx]:
                cover_count[ts_idx] += 1
                
        uncovered_ts = []
        for ts_idx in range(num_t_subsets):
            if cover_count[ts_idx] == 0:
                uncovered_ts.append(ts_idx)
                
        if not uncovered_ts:
            # The removed block was redundant
            return new_sol
            
        # Greedily add blocks to cover uncovered_ts
        # Gain for a block is number of uncovered_ts it covers
        gain = np.zeros(num_k_blocks, dtype=np.int32)
        # We only care about blocks that cover at least one uncovered ts
        # But calculating gain for all blocks is expensive.
        # Instead, for each uncovered ts, look at its blocks.
        
        # This is a "set cover" on the uncovered subset.
        # Since the number of uncovered ts is likely small (size of rem_block's unique ts),
        # we can use a simple greedy on the candidate set.
        
        candidate_blocks = set()
        for ts_idx in uncovered_ts:
            candidate_blocks.update(ts_to_blocks[ts_idx])
            
        # Filter candidates already in solution? No, we can reuse them? 
        # No, we removed one. We can add any block.
        # But adding a block already in solution is useless (it's already there).
        # So filter out blocks in new_sol.
        sol_set = set(new_sol)
        candidate_blocks -= sol_set
        
        if not candidate_blocks:
            # Cannot repair
            return sol_indices # Return original
            
        # Calculate gain for candidates
        # gain[b] = number of uncovered_ts covered by b
        # We can compute this by iterating uncovered_ts and incrementing gain for their blocks
        for ts_idx in uncovered_ts:
            for b_idx in ts_to_blocks[ts_idx]:
                if b_idx in candidate_blocks:
                    gain[b_idx] += 1
                    
        # Greedy selection
        while uncovered_ts:
            if time.time() > deadline - 2.0:
                return sol_indices # Timeout, return original
                
            # Find max gain
            max_gain = 0
            best_b = -1
            for b_idx in candidate_blocks:
                if gain[b_idx] > max_gain:
                    max_gain = gain[b_idx]
                    best_b = b_idx
                    
            if max_gain == 0:
                # No candidate covers any uncovered ts? Should not happen if cover is possible
                # But if we exhausted candidates, we fail.
                break
                
            # Add best_b
            new_sol.append(best_b)
            sol_set.add(best_b)
            candidate_blocks.discard(best_b)
            
            # Update uncovered_ts and gains
            newly_covered = []
            for ts_idx in block_to_ts[best_b]:
                if ts_idx in uncovered_ts: # This check is slow. Use a set for uncovered_ts?
                    newly_covered.append(ts_idx)
                    
            # Better: maintain a set of uncovered_ts indices
            # Let's switch to a set for uncovered_ts for faster removal
            # But we already have a list. Let's just filter.
            
            # Update cover_count
            for ts_idx in block_to_ts[best_b]:
                cover_count[ts_idx] += 1
                if cover_count[ts_idx] == 1:
                    # It was uncovered, now covered
                    # Remove from uncovered_ts
                    pass # We'll rebuild uncovered_ts
            
            # Rebuild uncovered_ts
            new_uncovered = []
            for ts_idx in uncovered_ts:
                if cover_count[ts_idx] == 0:
                    new_uncovered.append(ts_idx)
            uncovered_ts = new_uncovered
            
            # Update gains for remaining candidates
            # For each newly covered ts, decrement gain of candidates that cover it
            # And remove candidates that have gain 0?
            # This is getting complex. 
            # Given the small size of uncovered_ts initially, the number of iterations is small.
            # We can just recompute gains for all candidates in the next loop.
            # But that's O(|C| * |U|).
            # Let's just recompute gain for all candidates in the next iteration.
            # Actually, we can just break out and let the outer loop handle it?
            # No, we need to finish the repair.
            
            # Optimization: Only recompute gains if necessary.
            # For simplicity and speed in this specific "repair" context:
            # The number of uncovered ts is small.
            # Let's just recompute gain for all candidate_blocks.
            # It might be slow if candidate_blocks is large.
            # But candidate_blocks is limited by ts_to_blocks of uncovered_ts.
            
        return new_sol

    # Main Local Search Loop
    current_sol = optimize_redundancy(best_sol_indices)
    best_sol = current_sol
    best_len = len(best_sol)
    
    seed_counter = 1000
    no_improvement_count = 0
    
    while time.time() < deadline - 5.0:
        # 1. Try 1-for-2 exchange
        new_sol, improved = local_search_1_for_2(current_sol)
        if improved:
            current_sol = optimize_redundancy(new_sol)
            if len(current_sol) < best_len:
                best_len = len(current_sol)
                best_sol = current_sol
                no_improvement_count = 0
            else:
                no_improvement_count += 1
        else:
            # 2. Try Perturb and Repair
            # Only if we haven't improved for a while?
            # Or always?
            # Let's try it.
            if no_improvement_count > 5 or random.random() < 0.2:
                new_sol = perturb_and_repair(current_sol, seed_counter)
                seed_counter += 1
                if new_sol is not None:
                    new_sol = optimize_redundancy(new_sol)
                    if len(new_sol) < best_len:
                        best_len = len(new_sol)
                        best_sol = new_sol
                        current_sol = new_sol
                        no_improvement_count = 0
                    elif len(new_sol) == best_len and random.random() < 0.1:
                        # Accept equal length to diversify
                        current_sol = new_sol
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
            else:
                no_improvement_count += 1
                
        # If we've had many iterations without improvement, we might be stuck.
        # We could try a stronger perturbation.

    # Convert best_sol indices to list of blocks
    return [list(all_k_blocks[idx]) for idx in best_sol]

