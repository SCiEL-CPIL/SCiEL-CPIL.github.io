import itertools
import random
import time
import sys

# Precomputation for specific v, k, t is possible but construct is generic.
# We will do precomputation inside construct if v, k, t match the known hard case or are small enough.
# Given the 120s limit, we can afford significant precomputation for v=19.

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # If the problem is too large for this specific heuristic, fall back to simple greedy
    # But for v=19, k=8, t=5, we can be specific.
    
    # Generate all t-subsets
    t_subs = list(itertools.combinations(range(v), t))
    num_t = len(t_subs)
    t_sub_index = {ts: i for i, ts in enumerate(t_subs)}
    
    # Generate all k-subsets (blocks)
    k_subs = list(itertools.combinations(range(v), k))
    num_k = len(k_subs)
    
    # Precompute for each block, which t-subsets it covers
    # block_t_indices[i] is a list of indices into t_subs
    block_t_indices = []
    for i, block in enumerate(k_subs):
        covered = []
        for ts in itertools.combinations(block, t):
            covered.append(t_sub_index[ts])
        block_t_indices.append(covered)
        
    # Map from t-subset index to list of block indices that cover it
    # This is useful for the "pick block containing T" greedy step
    t_to_blocks = [[] for _ in range(num_t)]
    for i, block in enumerate(k_subs):
        for ts_idx in block_t_indices[i]:
            t_to_blocks[ts_idx].append(i)
            
    # --- GREEDY PHASE ---
    # uncovered: set of t-subset indices not yet covered
    uncovered = set(range(num_t))
    blocks_in_solution = set()
    solution_list = []
    
    # We pick blocks one by one
    while uncovered and time.time() < deadline - 5.0:
        # Pick a random uncovered t-subset to anchor the block selection
        # Picking the "hardest" one (least covered by remaining candidates) is better, 
        # but random is faster. Let's pick one that has fewest candidate blocks.
        # Actually, just picking any is fine for greedy.
        target_t_idx = next(iter(uncovered))
        
        candidate_blocks = t_to_blocks[target_t_idx]
        
        best_block_idx = -1
        best_gain = -1
        
        for b_idx in candidate_blocks:
            if b_idx in blocks_in_solution:
                continue
            # Calculate gain: number of uncovered t-subsets covered by this block
            gain = 0
            for ts_idx in block_t_indices[b_idx]:
                if ts_idx in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
                
        if best_block_idx == -1:
            # Should not happen if uncovered is not empty
            break
            
        # Add block
        blocks_in_solution.add(best_block_idx)
        solution_list.append(best_block_idx)
        
        # Update uncovered
        for ts_idx in block_t_indices[best_block_idx]:
            if ts_idx in uncovered:
                uncovered.discard(ts_idx)
                
    if not uncovered:
        # We have a valid solution
        pass
    else:
        # Fallback: simple greedy if the above failed or timed out
        # But the above is robust. If time runs out, we return what we have IF it's valid.
        # If not valid, we must ensure validity.
        # The loop condition `while uncovered` ensures we only stop if uncovered is empty or time runs out.
        # If time runs out with uncovered non-empty, the solution is INVALID.
        # We need a fallback that guarantees validity.
        # The simple greedy from the seed is valid.
        # Let's just run the seed's greedy if we are in trouble.
        pass

    # Check if we have a valid solution
    if uncovered:
        # Run a fast valid greedy (seed style) to ensure validity
        # This might be slower but guarantees a valid output
        uncovered_set = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered_set and time.time() < deadline - 1.0:
            block = list(min(uncovered_set))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered_set)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            blocks.append(block)
            uncovered_set.difference_update(itertools.combinations(block, t))
        if uncovered_set:
            # Extreme fallback: return all blocks? No, too many.
            # Just return the blocks we have if they cover everything? 
            # If uncovered_set is not empty, we failed.
            # For the purpose of this task, we assume the greedy above works.
            # But to be safe, let's ensure we return a valid list.
            # If we are here, we are in a bad state.
            # Let's just return the blocks from the precomputed greedy if we had stored them.
            # We didn't. Let's restart with a simpler valid greedy.
            pass

    # If we reached here with `uncovered` empty, we have a valid solution in `solution_list` (indices)
    # Convert to actual blocks
    current_blocks = [list(k_subs[i]) for i in solution_list]
    
    # --- LOCAL SEARCH PHASE ---
    # Maintain count of coverage for each t-subset
    cnt = [0] * num_t
    for i in solution_list:
        for ts_idx in block_t_indices[i]:
            cnt[ts_idx] += 1
            
    # Verify validity
    if any(c == 0 for c in cnt):
        # Should not happen if greedy worked
        pass

    # Local Search: Deletion and Swap
    # We will use the indices of the blocks in solution_list
    # To make deletion/swap fast, we use a set for membership
    sol_set = set(solution_list)
    
    # Shuffle solution_list for random access in swaps
    random.shuffle(solution_list)
    
    last_improvement = time.time()
    
    while time.time() < deadline - 2.0:
        # 1. Try Deletions
        deleted = False
        # Iterate through a copy or indices
        # We can iterate through the list
        for i in range(len(solution_list)):
            if time.time() > deadline - 2.0:
                break
            b_idx = solution_list[i]
            # Check if all t-subsets in this block are covered by >1 block
            can_delete = True
            for ts_idx in block_t_indices[b_idx]:
                if cnt[ts_idx] == 1:
                    can_delete = False
                    break
            if can_delete:
                # Delete
                for ts_idx in block_t_indices[b_idx]:
                    cnt[ts_idx] -= 1
                solution_list.pop(i)
                sol_set.remove(b_idx)
                deleted = True
                last_improvement = time.time()
                # Restart loop or continue?
                # Since indices might shift, restarting is safer but slower.
                # Given small N, restart is fine.
                break
                
        if deleted:
            continue
            
        # 2. Try Swaps
        # Pick a random block from solution
        if len(solution_list) > 0:
            pos = random.randint(0, len(solution_list) - 1)
            old_b_idx = solution_list[pos]
            
            # Pick a random block NOT in solution
            # We can pick a random block index and check membership
            # To be efficient, we can pre-generate a list of all blocks and pick from it
            # Or just pick random indices from range(num_k)
            
            swap_performed = False
            # Try a few random swaps
            for _ in range(10):
                if time.time() > deadline - 2.0:
                    break
                new_b_idx = random.randint(0, num_k - 1)
                if new_b_idx in sol_set:
                    continue
                    
                # Check if swap is valid
                # Condition: For any t-subset covered by old_b_idx with cnt==1,
                # it must also be covered by new_b_idx.
                
                # Get lists
                old_ts = block_t_indices[old_b_idx]
                new_ts = block_t_indices[new_b_idx]
                
                # Create a set of new_ts for O(1) lookup? 
                # Or just check intersection.
                # new_ts_set = set(new_ts)
                
                valid = True
                for ts_idx in old_ts:
                    if cnt[ts_idx] == 1:
                        # This t-subset will lose coverage from old_b_idx
                        # It MUST be covered by new_b_idx
                        # Check if ts_idx is in new_ts
                        # Since lists are small (56), linear search is ok, but set is better
                        # Let's use a set for new_ts for this check
                        if ts_idx not in new_ts_set: # Need to create new_ts_set
                             pass
                        # Optimization: Precompute sets? 
                        # 75k * 56 is memory heavy but ok? 
                        # Let's just do a simple check.
                        found = False
                        for n_ts in new_ts:
                            if n_ts == ts_idx:
                                found = True
                                break
                        if not found:
                            valid = False
                            break
                            
                if valid:
                    # Perform Swap
                    # Decrement old
                    for ts_idx in old_ts:
                        cnt[ts_idx] -= 1
                    # Increment new
                    for ts_idx in new_ts:
                        cnt[ts_idx] += 1
                        
                    solution_list[pos] = new_b_idx
                    sol_set.remove(old_b_idx)
                    sol_set.add(new_b_idx)
                    swap_performed = True
                    last_improvement = time.time()
                    break
                    
            if swap_performed:
                continue
                
        # 3. If no deletion or swap, try a perturbation?
        # Add a random block, then try to delete 2?
        # This is complex. Let's just keep trying swaps.
        
    # Final Result
    # Convert indices back to blocks
    final_blocks = [list(k_subs[i]) for i in solution_list]
    
    # Verify one last time
    # (Optional, but good for safety)
    # ...
    
    return final_blocks

# The above code has a bug in the swap check: `new_ts_set` is not defined.
# I will rewrite the code cleanly.

