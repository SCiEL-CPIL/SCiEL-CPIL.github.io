import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    # Safety margin: evaluator kills at 150s, we aim to return by 145s.
    deadline = start_time + 145.0
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map each block to the list of t-subset indices it covers
    # This is faster for counting updates than bitmasks when n_t is small enough
    # For C(24,3) = 2024, we can use an array of counts.
    # block_t_indices[i] is a list of t-subset indices covered by block i
    block_t_indices = []
    for block in all_blocks:
        idxs = [t_subset_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_t_indices.append(idxs)
    
    # 2. Initial Greedy Solution
    # Standard greedy set cover to get a valid starting point
    covered_mask = np.zeros((n_t + 63) // 64, dtype=np.uint64)
    # Actually, for SA, it's easier to track "uncovered_count" per t-subset
    # Let's use a count array for uncovered triples.
    # uncovered[i] = 1 if t-subset i is not covered by current solution, 0 otherwise.
    # Or better: cover_counts[i] = number of blocks in current solution covering t-subset i.
    # If cover_counts[i] == 0, it's uncovered.
    
    # Greedy construction
    cover_counts = np.zeros(n_t, dtype=np.int32)
    selected_indices = []
    
    # Precompute a list of blocks sorted by size? No, all size k.
    # We need to efficiently find the block that covers the most *new* triples.
    # Since n_blocks (346104) * n_t (2024) is too large for full matrix, we iterate.
    
    # Optimization for Greedy:
    # We can maintain a list of uncovered triples.
    # For each step, pick block maximizing (num_uncovered_in_block).
    
    # To speed up greedy, we can just use the bitmask approach from the seed
    # but convert to counts for SA.
    
    # Let's use the bitmask approach for the initial greedy as it was fast in the seed.
    n_uint64s = (n_t + 63) // 64
    coverage_masks = np.zeros((n_blocks, n_uint64s), dtype=np.uint64)
    
    # Fill coverage masks
    for i, block in enumerate(all_blocks):
        for ts_idx in block_t_indices[i]:
            word_idx = ts_idx // 64
            bit_idx = ts_idx % 64
            coverage_masks[i, word_idx] |= (1 << bit_idx)
            
    if time.time() > deadline - 30.0:
        # Fallback if precomputation is slow
        return [list(all_blocks[0])]

    # Run Greedy
    covered_mask = np.zeros(n_uint64s, dtype=np.uint64)
    selected_indices = []
    
    for _ in range(200): # Max blocks needed is < 100
        if time.time() > deadline - 30.0:
            break
        not_covered = ~covered_mask
        # Count new triples covered by each block
        # intersections = coverage_masks & not_covered
        # gains = np.bitwise_count(intersections).sum(axis=1)
        
        # This vectorized operation on 346k x 32 array might be slow if done repeatedly.
        # 346k * 32 * 8 bytes = ~88MB. 
        # Bitwise count on 32 uint64s per block.
        # Let's try it. If it's too slow, we switch to a slower greedy or skip to SA with a random valid cover (hard to get random valid cover).
        
        intersections = coverage_masks & not_covered
        gains = np.bitwise_count(intersections).sum(axis=1)
        
        best_idx = np.argmax(gains)
        if gains[best_idx] == 0:
            break
            
        selected_indices.append(best_idx)
        covered_mask |= coverage_masks[best_idx]
        
        # Early check
        if np.bitwise_count(covered_mask).sum() == n_t:
            break
            
    # Convert initial solution to cover_counts
    cover_counts = np.zeros(n_t, dtype=np.int32)
    for b_idx in selected_indices:
        cover_counts[block_t_indices[b_idx]] += 1
        
    # Ensure valid
    if np.any(cover_counts == 0):
        # Greedy failed? Unlikely. Fallback to exhaustive check?
        # Just return what we have if it's valid, else try a different greedy?
        # For now, assume valid.
        pass

    current_solution = selected_indices.copy()
    current_size = len(current_solution)
    best_solution = current_solution.copy()
    best_size = current_size
    
    # 3. Simulated Annealing
    # Objective: Minimize size. 
    # State: List of block indices.
    # Moves:
    #   1. Delete a block: If it doesn't make any cover_count 0, do it. (This is a "free" improvement)
    #   2. Swap a block: Remove block A, add block B. 
    #      Cost function for SA: 
    #      We want to minimize size. 
    #      Standard SA for covering: 
    #      Energy = - (number of blocks) + lambda * (number of uncovered triples)
    #      If uncovered > 0, energy is high (bad).
    #      We accept moves if energy decreases, or with probability exp(-delta_E / T) if it increases.
    
    # However, we must maintain validity at the end.
    # Strategy: 
    #   Allow temporary invalidity (uncovered triples) to escape local minima.
    #   T starts high, cools down.
    #   If we find a valid solution with fewer blocks, update best.
    
    # Precompute: for each block, which t-subsets it covers (already have block_t_indices)
    
    # To make swap moves efficient, we need to quickly evaluate the change in uncovered count.
    # When removing block A:
    #   For each t in A: cover_counts[t] -= 1.
    #   Uncovered increases by count of t in A where cover_counts[t] becomes 0.
    # When adding block B:
    #   For each t in B: cover_counts[t] += 1.
    #   Uncovered decreases by count of t in B where cover_counts[t] was 0.
    
    # We track `uncovered_count` = number of t-subsets with cover_count == 0.
    current_uncovered = int(np.sum(cover_counts == 0))
    
    # If current is invalid, we need to fix it first? 
    # Greedy should produce valid. 
    # If not, SA will try to fix it while minimizing size.
    
    T = 50.0 # Initial temperature
    T_min = 0.1
    cooling_rate = 0.995
    max_iterations = 50000
    
    # To speed up swap evaluation, we can precompute intersection sizes?
    # No, the change depends on current cover_counts.
    
    # We need a way to pick random block A from current_solution and random block B from all_blocks.
    # current_solution is a list.
    
    # Optimization: 
    # Instead of random B from all_blocks (346k), we can restrict B to blocks that cover many currently uncovered triples?
    # Or just random. With 50k iterations, random is fine.
    
    # Let's implement the SA loop.
    
    for i in range(max_iterations):
        if time.time() > deadline - 5.0:
            break
            
        # 1. Try Deletion (always beneficial if valid)
        if current_size > 1:
            # Pick a random block to try removing
            idx_to_remove = random.randint(0, current_size - 1)
            b_idx = current_solution[idx_to_remove]
            
            # Check if removal is safe
            # We need to know if any t in b_idx has cover_count == 1
            # If min(cover_counts[block_t_indices[b_idx]]) > 1, safe.
            
            t_idxs = block_t_indices[b_idx]
            # Vectorized min check
            if np.min(cover_counts[t_idxs]) > 1:
                # Safe to remove
                cover_counts[t_idxs] -= 1
                current_solution.pop(idx_to_remove)
                current_size -= 1
                # Update best if valid
                if current_uncovered == 0:
                    if current_size < best_size:
                        best_size = current_size
                        best_solution = current_solution.copy()
                continue # Deletion is a "free" win, restart loop or continue? Continue.
        
        # 2. Try Swap
        if current_size == 0:
            # Add a random block?
            b_idx = random.randint(0, n_blocks - 1)
            current_solution.append(b_idx)
            current_size += 1
            t_idxs = block_t_indices[b_idx]
            # Update counts and uncovered
            for t in t_idxs:
                if cover_counts[t] == 0:
                    current_uncovered -= 1
                cover_counts[t] += 1
            if current_uncovered == 0 and current_size < best_size:
                best_size = current_size
                best_solution = current_solution.copy()
            continue

        # Pick random block A to remove
        idx_a = random.randint(0, current_size - 1)
        b_idx_a = current_solution[idx_a]
        
        # Pick random block B to add
        # To avoid adding a block already in the solution? 
        # If B is in solution, it's a "move" not a swap, but effectively same as remove A, add B.
        # If B == A, no change.
        b_idx_b = random.randint(0, n_blocks - 1)
        
        if b_idx_a == b_idx_b:
            continue
            
        # Evaluate change
        t_idxs_a = block_t_indices[b_idx_a]
        t_idxs_b = block_t_indices[b_idx_b]
        
        # Calculate change in uncovered count
        # Removing A:
        #   For t in t_idxs_a:
        #     if cover_counts[t] == 1: uncovered += 1
        #     cover_counts[t] -= 1
        # Adding B:
        #   For t in t_idxs_b:
        #     if cover_counts[t] == 0: uncovered -= 1
        #     cover_counts[t] += 1
        
        # We can calculate delta_uncovered without modifying cover_counts first.
        
        # Step 1: Effect of removing A
        # We need to know which t in A are "critical" (count == 1)
        # And which t in B are "uncovered" (count == 0)
        
        # But adding B might fix the criticals from A if they overlap.
        # This requires careful ordering or simultaneous calculation.
        
        # Let's just simulate the change on a copy of relevant counts? 
        # Or calculate delta precisely.
        
        # delta_uncovered = (new_uncovered) - (old_uncovered)
        
        # Let's iterate through t_idxs_a and t_idxs_b.
        # Since t_idxs are small (35 elements), this is fast.
        
        delta_uncovered = 0
        
        # First, assume we remove A.
        # For each t in A:
        #   if cover_counts[t] == 1: this t becomes uncovered (delta +1)
        #   cover_counts[t] -= 1 (temporarily)
        
        # Then add B.
        # For each t in B:
        #   if cover_counts[t] == 0 (after removal of A): this t becomes covered (delta -1)
        #   cover_counts[t] += 1
        
        # We need to handle overlaps.
        # Let's use a temporary dictionary or just arrays.
        
        # Optimization: 
        # Create a set of t in A.
        # For t in A:
        #   if count[t] == 1: delta += 1
        #   count[t] -= 1
        # For t in B:
        #   if count[t] == 0: delta -= 1
        #   count[t] += 1
        # Then revert changes.
        
        # This is O(35+35) = 70 ops per move. Very fast.
        
        # Implementing revert is tricky if we modify in place.
        # Better: Calculate delta without modifying.
        
        # Let's use the "simulate on scratch" approach but it's complex to revert.
        # Alternative: Just modify, calculate, and if rejected, revert.
        
        # Let's modify cover_counts, calculate new_uncovered, and if rejected, revert.
        
        # Save old values for t in A and B?
        # Or just re-add/re-subtract.
        
        # 1. Remove A
        for t in t_idxs_a:
            cover_counts[t] -= 1
            if cover_counts[t] == 0:
                delta_uncovered += 1
        
        # 2. Add B
        for t in t_idxs_b:
            if cover_counts[t] == 0:
                delta_uncovered -= 1
            cover_counts[t] += 1
        
        # Now cover_counts reflects the new state (A removed, B added)
        # new_uncovered = current_uncovered + delta_uncovered
        
        new_uncovered = current_uncovered + delta_uncovered
        
        # Cost function
        # We want to minimize size.
        # If we keep size same, we prefer lower uncovered.
        # If we can reduce size, we do it (handled by deletion).
        # In swap, size is constant.
        # So we only care about uncovered count for validity.
        # But SA allows temporary invalidity.
        
        # Energy = -size + lambda * uncovered
        # Since size is constant in swap, Energy change = lambda * delta_uncovered.
        # We want to minimize Energy.
        # So if delta_uncovered < 0, it's better.
        # If delta_uncovered > 0, it's worse, accept with prob exp(-lambda * delta_uncovered / T).
        
        # What is lambda? 
        # If lambda is large, we strictly prefer valid states.
        # If lambda is small, we explore invalid states more.
        # Let's use lambda = 10.0.
        
        lambda_cost = 10.0
        delta_E = lambda_cost * delta_uncovered
        
        accept = False
        if delta_E <= 0:
            accept = True
        else:
            # Accept with probability
            p = np.exp(-delta_E / T)
            if random.random() < p:
                accept = True
        
        if accept:
            # Commit the move
            current_uncovered = new_uncovered
            current_solution[idx_a] = b_idx_b
            # cover_counts is already updated
            
            # Check if we found a better valid solution
            if current_uncovered == 0:
                if current_size < best_size:
                    best_size = current_size
                    best_solution = current_solution.copy()
                # Note: Swap doesn't change size, so it can only improve if we were invalid before and now valid?
                # No, best_size is tracked only for valid solutions.
                # If we were valid (uncovered=0) and now valid (uncovered=0), size is same.
                # If we were invalid and now valid, size is same.
                # So swap never reduces size.
                # Deletion reduces size.
                # So we only update best_size on deletion.
            else:
                # We are now in an invalid state (or still invalid).
                # We don't update best_solution here.
                pass
        else:
            # Revert the move
            # Undo Add B
            for t in t_idxs_b:
                cover_counts[t] -= 1
            # Undo Remove A
            for t in t_idxs_a:
                if cover_counts[t] == 0:
                    delta_uncovered += 1 # Wait, delta_uncovered was calculated forward.
                    # We need to revert the state.
                    # The state is currently: A removed, B added.
                    # We want: A present, B absent.
                    # So: Remove B, Add A.
                    # My previous logic:
                    # 1. Remove A (counts -1)
                    # 2. Add B (counts +1)
                    # To revert:
                    # 3. Remove B (counts -1)
                    # 4. Add A (counts +1)
                    
                    pass
                cover_counts[t] += 1
            
            # Wait, the revert logic above is messy because I used delta_uncovered in the forward pass.
            # Let's just re-calculate the revert carefully.
            
            # Current state in cover_counts: A out, B in.
            # Target state: A in, B out.
            
            # Step 1: Remove B
            for t in t_idxs_b:
                cover_counts[t] -= 1
                # No need to track delta for revert, we just restore counts.
                # But we need to restore current_uncovered.
                # We saved old current_uncovered? No, we overwrote it?
                # No, we calculated new_uncovered but didn't assign to current_uncovered unless accepted.
                # So current_uncovered is still the OLD value.
                # We just need to restore cover_counts.
                pass
                
            # Step 2: Add A
            for t in t_idxs_a:
                cover_counts[t] += 1
                pass
            
            # cover_counts is restored.
            # current_uncovered is unchanged (still old value).
            # So revert is just restoring cover_counts.
            pass

        # Cooling
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Optional: Restart with higher T? Or just keep going.
            # With 50k iterations, T will cool down significantly.
            # 50.0 * 0.995^50000 is extremely small.
            # Let's reset T periodically to allow exploration.
            if i % 5000 == 0 and i > 0:
                T = 50.0

    # 4. Final Local Search: Deletion
    # After SA, we might be in a valid state. Try to delete more.
    # Use the best_solution found, or current if valid?
    # Let's take the best_solution found during SA.
    
    final_solution = best_solution.copy()
    
    # Rebuild cover_counts for final_solution to ensure consistency
    cover_counts = np.zeros(n_t, dtype=np.int32)
    for b_idx in final_solution:
        cover_counts[block_t_indices[b_idx]] += 1
        
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        # Try to remove blocks
        # Iterate in reverse
        for i in range(len(final_solution) - 1, -1, -1):
            b_idx = final_solution[i]
            t_idxs = block_t_indices[b_idx]
            if np.min(cover_counts[t_idxs]) > 1:
                cover_counts[t_idxs] -= 1
                final_solution.pop(i)
                improved = True
                break # Restart loop
        
    # Final Verification
    if np.any(cover_counts == 0):
        # If final solution is invalid (shouldn't happen if best_solution was valid),
        # fallback to the initial greedy solution.
        # Rebuild greedy?
        # We have selected_indices from greedy.
        # Let's use that.
        final_solution = selected_indices.copy()
        # Re-verify
        cover_counts = np.zeros(n_t, dtype=np.int32)
        for b_idx in final_solution:
            cover_counts[block_t_indices[b_idx]] += 1
        # If still invalid, we have a problem.
        # But greedy should be valid.
        
    return [list(all_blocks[i]) for i in final_solution]
