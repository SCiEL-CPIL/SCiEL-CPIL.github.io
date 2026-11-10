import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    # 1. Precomputation
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    all_t_subsets = list(itertools.combinations(points, t))
    n_t_subsets = len(all_t_subsets)
    
    # Map t-subset to index
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # block_subsets[i] = list of indices of t-subsets covered by block i
    block_subsets = [[] for _ in range(n_blocks)]
    # subset_to_blocks[j] = list of block indices covering t-subset j
    subset_to_blocks = [[] for _ in range(n_t_subsets)]
    
    for i, block in enumerate(all_blocks):
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            block_subsets[i].append(idx)
            subset_to_blocks[idx].append(i)
            
    # 2. Greedy Construction
    in_block = [False] * n_blocks
    cover_count = [0] * n_t_subsets
    uncovered_count = n_t_subsets
    solution = []
    
    while uncovered_count > 0:
        best_block = -1
        max_gain = 0
        # Iterate all blocks to find max gain
        # Optimization: We can break early if gain == uncovered_count? Unlikely.
        for i in range(n_blocks):
            if in_block[i]:
                continue
            gain = 0
            for s_idx in block_subsets[i]:
                if cover_count[s_idx] == 0:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_block = i
                if gain == uncovered_count: # Can't get better
                    break
        
        if best_block == -1 or max_gain == 0:
            # Should not happen if problem is solvable
            break
            
        # Add block
        in_block[best_block] = True
        solution.append(best_block)
        for s_idx in block_subsets[best_block]:
            if cover_count[s_idx] == 0:
                uncovered_count -= 1
            cover_count[s_idx] += 1
            
    # 3. Local Search / Refinement
    # Helper: Remove redundant blocks
    def remove_redundants():
        nonlocal in_block, cover_count, solution
        changed = True
        while changed:
            changed = False
            # Iterate over a copy of solution to allow removal
            for b in list(solution):
                if not in_block[b]:
                    continue
                # Check if b is redundant
                redundant = True
                for s_idx in block_subsets[b]:
                    if cover_count[s_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove b
                    in_block[b] = False
                    solution.remove(b)
                    for s_idx in block_subsets[b]:
                        cover_count[s_idx] -= 1
                    changed = True
                    # Restart loop as indices/counts changed
                    break # Break inner, restart outer while
    
    remove_redundants()
    
    best_count = len(solution)
    
    # Local Search Loop
    while time.time() < deadline:
        # Pick a random block from solution to try to replace
        if not solution:
            break
        b_out = random.choice(solution)
        
        # Find critical subsets for b_out
        criticals = []
        for s_idx in block_subsets[b_out]:
            if cover_count[s_idx] == 1:
                criticals.append(s_idx)
                
        if not criticals:
            # b_out is redundant, remove it
            in_block[b_out] = False
            solution.remove(b_out)
            for s_idx in block_subsets[b_out]:
                cover_count[s_idx] -= 1
            remove_redundants()
            continue
            
        # Find candidate b_in that covers all criticals
        # Intersection of subset_to_blocks for all criticals
        # Since criticals can be many, we can just check blocks in subset_to_blocks[first_critical]
        # and verify they cover the rest
        
        candidates = []
        first_crit = criticals[0]
        for b_in in subset_to_blocks[first_crit]:
            if in_block[b_in]:
                continue
            # Check if b_in covers all other criticals
            valid = True
            for s_idx in criticals[1:]:
                # Efficient check: is s_idx in block_subsets[b_in]?
                # We can use a set for block_subsets[b_in] for O(1) lookup, 
                # but we stored lists. 
                # For speed, let's precompute block_subsets as sets? 
                # 50k sets of 35 ints is fine.
                pass 
            
        # To make this faster, let's precompute block_subsets as sets or use a boolean matrix?
        # Boolean matrix 50k x 3876 is 190M bits = 24MB. Feasible.
        # Or just sets.
        
        # Let's refactor precomputation to use sets for block_subsets for O(1) lookup.
        # Actually, the loop above is inside the time loop. We need it fast.
        
        # Let's assume we precomputed block_subsets_sets.
        
        # If we find a valid b_in:
        # Swap
        # Update counts
        # Remove redundants
        # If len(solution) < best_count, update best
        
    # Return the best solution found
    # We need to store the best solution separately if we modify the current one?
    # The swap logic modifies in_block/cover_count/solution in place.
    # If a swap fails to reduce size, we might want to revert?
    # Or just accept any valid swap? 
    # Accepting any valid swap that maintains coverage is a random walk.
    # We should only keep the state if it leads to a smaller size or is "better" in some way.
    # Simpler: Keep track of best_solution (list of blocks) and best_count.
    # If current len(solution) < best_count, save it.
    
    # The code above modifies `solution` in place.
    # We should snapshot the best solution.
    
    # Let's refine the local search to be safer.
    
    return [list(all_blocks[i]) for i in solution]
