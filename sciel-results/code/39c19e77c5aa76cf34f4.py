import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin for 120s limit
    
    # 1. Precomputation
    # Generate all t-subsets and map them to IDs
    all_subsets = list(itertools.combinations(range(v), t))
    num_subsets = len(all_subsets)
    subset_to_id = {s: i for i, s in enumerate(all_subsets)}
    
    # Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute mask for each block
    # mask is an integer where bit i is set if the block covers subset i
    block_masks = [0] * num_blocks
    block_subset_indices = [[] for _ in range(num_blocks)] # Store indices for faster iteration in local search
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        indices = []
        # Iterate over t-subsets of this block
        for sub in itertools.combinations(block, t):
            s_id = subset_to_id[sub]
            mask |= (1 << s_id)
            indices.append(s_id)
        block_masks[b_idx] = mask
        block_subset_indices[b_idx] = indices

    # Helper to run greedy
    def run_greedy():
        uncovered_mask = (1 << num_subsets) - 1
        selected = []
        
        while uncovered_mask:
            best_gain = -1
            best_block = -1
            # Randomization: shuffle blocks or just pick first max? 
            # To allow diversity, we can randomize tie-breaking.
            # But for speed, linear scan.
            for b_idx in range(num_blocks):
                gain = (block_masks[b_idx] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                    # Early exit if gain is perfect? Max possible gain is min(popcount(uncovered), popcount(block_mask))
                    # Max popcount for a block is 35.
                    if best_gain == 35: 
                        # Can't get better than covering all subsets in the block
                        # But there might be another block with 35.
                        # We break to pick this one.
                        pass 
            
            if best_block == -1:
                break # Should not happen if uncovered_mask is not 0 and blocks cover all subsets
                
            selected.append(best_block)
            uncovered_mask &= ~block_masks[best_block]
            
        return selected

    # Helper to run local search on a given set of blocks
    def local_search(initial_blocks):
        selected = set(initial_blocks)
        # coverage_count[s] = number of blocks in selected covering s
        coverage_count = [0] * num_subsets
        
        # Initialize counts
        for b_idx in selected:
            for s_idx in block_subset_indices[b_idx]:
                coverage_count[s_idx] += 1
        
        current_size = len(selected)
        best_size = current_size
        best_solution = list(selected)
        
        no_improvement_count = 0
        max_no_improvement = 200 # Reset or restart logic if stuck
        
        while time.time() < deadline:
            improved = False
            
            # Try to remove a random block
            if selected:
                b_out = random.choice(list(selected))
                
                # Check if removal is valid
                # A block can be removed if for all s in block, coverage_count[s] > 1
                # i.e., min(coverage_count[s] for s in block) > 1
                removable = True
                for s_idx in block_subset_indices[b_out]:
                    if coverage_count[s_idx] == 1:
                        removable = False
                        break
                
                if removable:
                    # Remove it
                    selected.remove(b_out)
                    for s_idx in block_subset_indices[b_out]:
                        coverage_count[s_idx] -= 1
                    current_size -= 1
                    improved = True
                    if current_size < best_size:
                        best_size = current_size
                        best_solution = list(selected)
                    no_improvement_count = 0
                    continue # Try another removal immediately
                else:
                    # Try a swap
                    # Find critical subsets: those with count 1 covered by b_out
                    critical_subsets = []
                    for s_idx in block_subset_indices[b_out]:
                        if coverage_count[s_idx] == 1:
                            critical_subsets.append(s_idx)
                    
                    if critical_subsets:
                        # We need a block b_in that covers all critical_subsets
                        # And b_in not in selected
                        # And ideally covers other "low count" subsets to help future removals
                        best_in = -1
                        best_score = -1
                        
                        # To speed up, we can pre-compute for each block which critical sets it covers?
                        # No, just iterate all blocks not in selected.
                        # This is O(num_blocks) per swap attempt.
                        # 50k blocks. If we do this often, it might be slow.
                        # Optimization: Only try swaps if we are stuck?
                        # Or limit swap attempts.
                        
                        # Heuristic: Only try to find a swap if we haven't improved in a while
                        if no_improvement_count > 50:
                            for b_idx in range(num_blocks):
                                if b_idx in selected:
                                    continue
                                
                                # Check if b_idx covers all critical_subsets
                                covers_all = True
                                for s_idx in critical_subsets:
                                    if not (block_masks[b_idx] >> s_idx) & 1:
                                        covers_all = False
                                        break
                                
                                if not covers_all:
                                    continue
                                
                                # Score: number of subsets in b_idx with count == 1 (excluding critical ones which are already 1)
                                # Actually, we want to increase redundancy of other count-1 subsets.
                                # Let's count how many s in b_idx have count == 1.
                                # This makes them count 2, making them safer to remove from other blocks.
                                score = 0
                                for s_idx in block_subset_indices[b_idx]:
                                    if coverage_count[s_idx] == 1:
                                        score += 1
                                
                                if score > best_score:
                                    best_score = score
                                    best_in = b_idx
                                    
                            if best_in != -1:
                                # Perform swap
                                selected.remove(b_out)
                                for s_idx in block_subset_indices[b_out]:
                                    coverage_count[s_idx] -= 1
                                    
                                selected.add(best_in)
                                for s_idx in block_subset_indices[best_in]:
                                    coverage_count[s_idx] += 1
                                
                                # Size unchanged, so best_size doesn't update here directly.
                                # But structure changed.
                                improved = True # Mark as activity
                                no_improvement_count = 0
                                continue
            
            if not improved:
                no_improvement_count += 1
                if no_improvement_count > max_no_improvement:
                    break # Stuck, break to restart greedy
            
        return best_solution, best_size

    best_blocks_overall = None
    best_size_overall = float('inf')
    
    # Run multiple greedy + local search iterations
    # Greedy is fast, Local Search is the heavy part.
    # We can do a few greedy runs, then spend most time on local search of the best one.
    
    # 1. Run a few greedy passes to get good initial candidates
    candidates = []
    while time.time() < deadline - 20.0: # Leave 20s for local search
        sol = run_greedy()
        candidates.append(sol)
        if len(candidates) >= 5:
            break
            
    # 2. Apply local search to each candidate, or just the best one?
    # Let's pick the shortest candidate.
    if not candidates:
        return run_greedy() # Fallback
        
    candidates.sort(key=lambda x: len(x))
    
    # We can try local search on the top 1 or 2 candidates
    for sol in candidates[:1]: # Just the best one to maximize time for polish
        polished_sol, polished_size = local_search(sol)
        
        if polished_size < best_size_overall:
            best_size_overall = polished_size
            best_blocks_overall = polished_sol
            
        # If we have time, we could try to run local search on the next candidate
        # But local_search uses the global deadline, so it will stop naturally.
        # However, if the first one finishes early, we might want to try the second?
        # The local_search function breaks on time.
        # Let's just run it once. The while loop inside local_search handles the time.
        
    if best_blocks_overall is None:
        best_blocks_overall = candidates[0]
        
    # Convert to list of lists of ints as required
    return [list(block) for block in best_blocks_overall]

