import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0
    
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute all t-subsets for each block
    # This is crucial for fast gain/loss calculation
    block_t_subsets = []
    for b in all_blocks:
        block_t_subsets.append(list(itertools.combinations(b, t)))
    
    # Map t-subset to index for coverage counting
    all_t_subsets = list(itertools.combinations(points, t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Flatten t-subsets to integer indices for faster numpy operations if needed, 
    # but here we will use lists of indices for each block.
    block_t_indices = []
    for subs in block_t_subsets:
        idxs = [t_subset_to_idx[s] for s in subs]
        block_t_indices.append(np.array(idxs, dtype=np.int32))

    # --- Phase 1: Greedy Initialization ---
    # Standard greedy: pick block covering most uncovered t-subsets
    uncovered = np.ones(num_t_subsets, dtype=np.int8)
    selected_blocks = []
    
    # Precompute incidence for greedy? 
    # For v=19, k=7, t=4: C(19,7)=50388, C(19,4)=3876.
    # Matrix is 50388 x 3876. Sparse is fine.
    # But for SA, we need to update counts dynamically.
    # Greedy can be done with a priority queue or just max scan if fast enough.
    # 50388 blocks * 126 t-subsets per block = ~6.3M operations.
    
    # To speed up greedy, we can precompute the t-subset indices for each block.
    # We already have block_t_indices.
    
    # Simple greedy loop
    # We need to know how many new subsets a block covers.
    # new_count = sum(1 for idx in block_t_indices[b] if uncovered[idx] == 1)
    
    # This is O(126) per block check. 50388 * 126 is ~6M ops per iteration.
    # Number of iterations ~ 150. Total ~900M ops. Too slow in Python loop.
    
    # Optimization: Use numpy for the gain calculation?
    # uncovered is a vector.
    # For each block, gain = dot(uncovered, mask_block)
    # We can build a sparse matrix A (blocks x t_subsets) once.
    
    from scipy.sparse import csr_matrix
    
    row_indices = []
    col_indices = []
    data = []
    for i, idxs in enumerate(block_t_indices):
        for j in idxs:
            row_indices.append(i)
            col_indices.append(j)
            data.append(1)
            
    A = csr_matrix((data, (row_indices, col_indices)), shape=(num_blocks, num_t_subsets))
    
    # Greedy
    active = np.ones(num_t_subsets, dtype=np.float64)
    while True:
        if time.time() > deadline - 100.0:
            break
        gains = A.dot(active)
        if np.max(gains) == 0:
            break
        b_idx = np.argmax(gains)
        selected_blocks.append(b_idx)
        # Update active
        # Get non-zero indices of row b_idx
        covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
        active[covered] = 0
        if np.all(active == 0):
            break
            
    # Convert to set for faster lookup? No, we need list for SA.
    # We need to track which blocks are in the solution.
    # For SA, we will maintain:
    # 1. A set of block indices in the current solution.
    # 2. An array `cover_counts` of size num_t_subsets, counting how many blocks cover each t-subset.
    
    current_solution = set(selected_blocks)
    cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
    for b_idx in current_solution:
        covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
        cover_counts[covered] += 1
        
    # Check validity (should be valid)
    # If not, we might have a bug, but greedy ensures validity.
    
    best_solution = list(current_solution)
    best_size = len(best_solution)
    
    # --- Phase 2: Simulated Annealing ---
    # Move: Swap one block out, one block in.
    # Cost function: 
    # Let S be the set of blocks.
    # Let U be the set of uncovered t-subsets (cover_counts == 0).
    # Primary objective: Minimize |U| (number of uncovered subsets).
    # Secondary objective: Minimize |S| (number of blocks).
    # We can define a penalty for uncovered subsets.
    # Cost = |S| + lambda * |U|
    # If lambda is large enough, minimizing cost will prioritize covering all subsets.
    # However, we want a valid cover. 
    # Better: 
    # If |U| > 0, Cost = |U| (we must fix coverage first).
    # If |U| == 0, Cost = |S| (we can now reduce size).
    # This is a lexicographic objective.
    
    # Let's use a single scalar cost:
    # Cost = |S| + M * |U| where M is large.
    # But we need to be careful with swaps that increase |S| but decrease |U|.
    
    # Actually, a simpler approach for covering designs:
    # State: (S, cover_counts)
    # Move: Remove b_out, Add b_in.
    # Delta S = 0 (size stays same).
    # Delta U: 
    #   Subsets only covered by b_out: become uncovered (if cover_counts was 1).
    #   Subsets not covered by anyone: become covered (if cover_counts was 0 and b_in covers them).
    #   Subsets covered by others: no change in coverage status (count changes, but status 0->0 or >0->>0).
    
    # Let's track:
    # 1. num_uncovered = sum(cover_counts == 0)
    # 2. num_blocks = len(S)
    
    # We will perform swaps that keep num_blocks constant.
    # This allows us to "shuffle" blocks to find a configuration where some blocks become redundant (can be removed).
    # After SA, we do a deletion pass.
    
    # Cost for SA:
    # We want to minimize the number of uncovered subsets.
    # If we find a state with 0 uncovered, we then try to reduce size.
    
    # Let's define Cost = num_uncovered.
    # We accept moves that decrease num_uncovered.
    # We accept moves that increase num_uncovered with probability exp(-delta/T).
    # We do NOT change the number of blocks in SA.
    # After SA, we run a deletion pass.
    # We restart SA if we get stuck? Or just run for a fixed time.
    
    # Since we start with a valid cover (num_uncovered = 0), any swap that keeps num_uncovered = 0 is "neutral".
    # We need to explore the space of valid covers of fixed size to find one where a block is redundant.
    # If we swap b_out with b_in, and the result is still valid, we have a new valid cover.
    # If we find a valid cover where a block is redundant, we can remove it, reducing size by 1.
    # Then we restart SA with the new smaller size? Or continue?
    
    # Strategy:
    # 1. Start with greedy solution size N.
    # 2. Run SA on fixed-size N swaps to find "loose" covers.
    # 3. Check for redundant blocks. If found, remove one. N decreases by 1.
    # 4. Repeat until no more reduction or time runs out.
    
    # This is a "ratchet" approach.
    
    # To make SA effective at finding redundant blocks, we might need to allow temporary invalid states?
    # If we only allow valid states, we are moving on the surface of the valid solution manifold.
    # If the manifold is connected, we can find any valid solution of size N.
    # If there exists a valid solution of size N-1, it is a subset of some valid solution of size N?
    # Not necessarily. But if we can find ANY valid solution of size N that has a redundant block, we can reduce.
    
    # Let's implement the SA loop.
    
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.99
    
    # Precompute for fast delta calculation?
    # When removing b_out:
    #   For each t-subset s in b_out:
    #       cover_counts[s] -= 1
    #       if cover_counts[s] == 0: num_uncovered += 1
    # When adding b_in:
    #   For each t-subset s in b_in:
    #       if cover_counts[s] == 0: num_uncovered -= 1
    #       cover_counts[s] += 1
    
    # We need to know the intersection of b_out and b_in? No, just iterate over their t-subsets.
    # Each block has 126 t-subsets.
    # So a swap takes ~252 operations.
    # 150s / 252 ops ~ 600,000 swaps? That's very fast.
    # We can do many more.
    
    # Let's run SA for a fixed number of iterations or time.
    
    num_blocks_current = len(current_solution)
    list_solution = list(current_solution)
    
    # We need a list of blocks NOT in the solution to pick b_in from.
    # This changes as we swap.
    # Maintaining a set of available blocks is O(1) for check, O(1) for add/remove.
    
    available_blocks = set(range(num_blocks))
    for b in current_solution:
        available_blocks.remove(b)
        
    T = T_start
    iterations = 0
    max_iterations = 5000000 # High limit, controlled by time
    
    # We need to track best solution found during the whole process
    global_best_size = best_size
    global_best_solution = best_solution[:]
    
    while time.time() < deadline - 10.0:
        # Check for redundancy in current solution
        # If a block is redundant, remove it and restart SA with smaller size
        redundant_found = False
        for b_idx in list_solution:
            # Check if removing b_idx leaves all subsets covered
            # i.e., for all s in b_idx, cover_counts[s] > 1
            is_redundant = True
            covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
            for s in covered:
                if cover_counts[s] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                # Remove it
                current_solution.remove(b_idx)
                list_solution.remove(b_idx)
                available_blocks.add(b_idx)
                covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                cover_counts[covered] -= 1
                num_blocks_current -= 1
                redundant_found = True
                
                # Update global best
                if num_blocks_current < global_best_size:
                    global_best_size = num_blocks_current
                    global_best_solution = list_solution[:]
                    
                # Reset SA parameters?
                # We can restart SA with the new size.
                # For simplicity, let's just continue with the new state.
                # But we need to pick a new b_out from the new list.
                break
                
        if redundant_found:
            # Restart SA loop logic?
            # Actually, we can just continue the while loop.
            # But we should probably reset T to allow exploration again?
            # Or keep cooling?
            # Let's keep cooling, but if we get stuck, we might need to reheat.
            # For now, continue.
            pass
        
        # Perform a swap
        if not list_solution:
            break
            
        # Pick b_out randomly from current solution
        b_out = random.choice(list_solution)
        
        # Pick b_in randomly from available blocks
        # Using random.choice on a set is O(1) in Python 3?
        # Actually, random.choice requires a sequence.
        # We can maintain a list of available blocks?
        # Updating a list is O(N).
        # Let's use a list for available blocks too, and swap with the last element for removal.
        
        # Let's refactor to use lists for available blocks for O(1) random selection.
        # available_list = list(available_blocks)
        # This is expensive to build.
        
        # Alternative: Keep a set, and use `random.choice(list(available_blocks))`? No, too slow.
        # Keep a list `available_list` and `available_set`.
        # When removing b_in from available:
        #   idx = available_list.index(b_in) -> O(N)
        #   swap with last, pop.
        
        # Let's maintain `available_list` and `available_set`.
        # Initialization:
        # available_set = set(range(num_blocks)) - current_solution
        # available_list = list(available_set)
        
        # This requires re-initialization if we change the structure.
        # Let's do it.
        
        if 'available_list' not in locals():
             available_set = set(range(num_blocks)) - current_solution
             available_list = list(available_set)
             
        if not available_list:
            break
            
        b_in_idx = random.randint(0, len(available_list) - 1)
        b_in = available_list[b_in_idx]
        
        # Apply swap: Remove b_out, Add b_in
        # 1. Remove b_out from solution
        current_solution.remove(b_out)
        list_solution.remove(b_out)
        available_set.add(b_out)
        # Add b_out to available_list? 
        # We are swapping b_in out of available_list.
        # Let's swap b_in with last in available_list, then pop.
        # Then add b_out to available_list.
        
        # Remove b_in from available_list
        available_list[b_in_idx] = available_list[-1]
        available_list.pop()
        available_set.remove(b_in)
        
        # Add b_out to available_list
        available_list.append(b_out)
        
        # Update cover_counts
        delta_uncovered = 0
        
        # Remove b_out
        covered_out = A.indices[A.indptr[b_out]:A.indptr[b_out+1]]
        for s in covered_out:
            cover_counts[s] -= 1
            if cover_counts[s] == 0:
                delta_uncovered += 1
                
        # Add b_in
        covered_in = A.indices[A.indptr[b_in]:A.indptr[b_in+1]]
        for s in covered_in:
            if cover_counts[s] == 0:
                delta_uncovered -= 1
            cover_counts[s] += 1
            
        # Add b_in to solution
        current_solution.add(b_in)
        list_solution.append(b_in)
        
        # Check if move is valid (delta_uncovered == 0)
        # If delta_uncovered > 0, we increased uncovered count.
        # If delta_uncovered < 0, we decreased uncovered count (improvement).
        # Note: Since we start with 0 uncovered, delta_uncovered can only be >= 0?
        # Wait. If we remove b_out, some subsets might become uncovered (count 1 -> 0).
        # Then we add b_in. If b_in covers those subsets, count goes 0 -> 1.
        # So delta_uncovered is the net change.
        
        # Cost function:
        # We want to minimize uncovered.
        # If delta_uncovered > 0, it's a worse state (more uncovered).
        # If delta_uncovered == 0, it's neutral (still valid).
        # If delta_uncovered < 0, it's better (less uncovered, though we started at 0 so this shouldn't happen unless we allowed invalid states in previous steps).
        
        # Since we start with valid cover (0 uncovered), and we only swap,
        # the number of uncovered subsets can only increase or stay 0.
        # Wait, can it decrease?
        # If we have 0 uncovered, removing b_out makes some uncovered (count 1->0).
        # Adding b_in covers some. If b_in covers exactly the ones that became uncovered, delta is 0.
        # If b_in covers MORE than that, it covers some that were already covered (count >0 -> >1). No change in "uncovered" status.
        # So delta_uncovered >= 0 always if starting from valid.
        
        # So we are looking for swaps with delta_uncovered == 0.
        # These are "valid" swaps.
        # We accept all valid swaps (delta == 0).
        # We reject invalid swaps (delta > 0) unless we allow temperature?
        # If we allow temperature, we can explore invalid states to escape local minima.
        # But we need to be careful: if we go invalid, we must find a way back.
        
        # Let's allow invalid states with probability exp(-delta/T).
        
        if delta_uncovered > 0:
            # Calculate probability
            # delta is the increase in uncovered subsets.
            # We use delta as the "energy" difference.
            p = np.exp(-delta_uncovered / T)
            if random.random() > p:
                # Revert
                # Remove b_in, Add b_out
                current_solution.remove(b_in)
                list_solution.remove(b_in)
                available_set.add(b_in)
                available_list.append(b_in)
                
                # Remove b_out from available_list (it's at the end? No, we added it)
                # Wait, we added b_out to available_list.
                # We need to remove b_out from available_list and add b_in.
                # This is getting messy.
                
                # Let's simplify the revert logic.
                # State before swap:
                # Solution: S (with b_out, without b_in)
                # Available: A (with b_in, without b_out)
                
                # State after swap:
                # Solution: S' (with b_in, without b_out)
                # Available: A' (with b_out, without b_in)
                
                # To revert:
                # Remove b_in from Solution, Add b_out to Solution.
                # Remove b_out from Available, Add b_in to Available.
                
                current_solution.remove(b_in)
                list_solution.remove(b_in)
                current_solution.add(b_out)
                list_solution.append(b_out)
                
                # Update cover_counts back
                # Remove b_in
                for s in covered_in:
                    cover_counts[s] -= 1
                # Add b_out
                for s in covered_out:
                    cover_counts[s] += 1
                    
                # Update available structures
                available_set.remove(b_in)
                available_set.add(b_out)
                
                # Fix lists
                # b_in was at b_in_idx in available_list. We swapped it with last and popped.
                # So b_in is gone from list. b_out was added to end.
                # To revert:
                # Remove b_out from list (it's at end).
                available_list.pop()
                # Add b_in to list.
                available_list.append(b_in)
                
                # We don't need to restore the exact position of b_in, just the set membership.
                
                T *= cooling_rate
                iterations += 1
                continue
            
        # Accept move
        T *= cooling_rate
        iterations += 1
        
        # Optional: If we have been stuck in valid states for a long time,
        # we might want to force a "kick" by accepting a slightly worse state?
        # Or just let the temperature drop.
        
        # If T is very low, we only accept delta=0 moves.
        
    # Final Deletion Pass
    # We might have ended up in a valid state.
    # Check for redundant blocks again.
    changed = True
    while changed:
        changed = False
        to_remove = []
        for b_idx in list_solution:
            covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
            redundant = True
            for c in covered:
                if cover_counts[c] == 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(b_idx)
                
        if to_remove:
            for b_idx in to_remove:
                covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                cover_counts[covered] -= 1
                current_solution.remove(b_idx)
                list_solution.remove(b_idx)
                available_set.add(b_idx)
                available_list.append(b_idx)
            changed = True
            if len(list_solution) < global_best_size:
                global_best_size = len(list_solution)
                global_best_solution = list_solution[:]

    # Return the best solution found
    # Ensure we return the global best, not just the current one
    # If global_best_solution is valid, use it.
    # We should verify global_best_solution just in case.
    
    # For simplicity and safety, if the current solution is valid, use it.
    # If not, use global_best_solution.
    
    # Check current validity
    if np.all(cover_counts >= 1):
        final_blocks = list_solution
    else:
        final_blocks = global_best_solution
        
    # Convert to list of lists
    blocks = [list(all_blocks[b_idx]) for b_idx in final_blocks]
    
    # Final sanity check
    # If something went wrong and blocks is empty or invalid, fallback to greedy?
    # The greedy solution was valid.
    # If final_blocks is invalid, we should return the greedy one.
    
    # Let's verify final_blocks
    cover_check = np.zeros(num_t_subsets, dtype=np.int32)
    for b_idx in final_blocks:
        covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
        cover_check[covered] += 1
        
    if not np.all(cover_check >= 1):
        # Fallback to greedy
        final_blocks = selected_blocks
        blocks = [list(all_blocks[b_idx]) for b_idx in final_blocks]
        
    return blocks
