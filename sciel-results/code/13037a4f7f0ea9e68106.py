import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    block_idx_map = {b: i for i, b in enumerate(all_blocks)}
    
    # Precompute coverage matrix: M[i, j] = 1 if block i covers t-subset j
    # n_blocks ~ 75k, n_t ~ 11.6k. 75k * 11.6k * 8 bytes ~ 7 GB. Too big for RAM.
    # We need a sparse or compressed representation or a different approach.
    # However, for v=19, k=8, t=5:
    # n_blocks = C(19,8) = 75582
    # n_t = C(19,5) = 11628
    # Each block covers C(8,5)=56 t-subsets.
    # Total non-zeros = 75582 * 56 = 4.2M.
    # We can store this as a sparse matrix or list of lists.
    # For fast coverage count updates, we need to add/subtract rows.
    # NumPy sparse is slow. Let's use a dense int8 array if possible?
    # 75582 * 11628 is too big for dense.
    
    # Alternative: Store for each block, the list of t-subset indices it covers.
    # And maintain a global count array of size n_t.
    # When adding a block, increment counts for its 56 subsets.
    # When removing, decrement.
    # This is O(56) per move. Very fast.
    
    block_t_indices = []
    for b in all_blocks:
        subs = list(itertools.combinations(b, t))
        indices = [t_sub_to_idx[sub] for sub in subs]
        block_t_indices.append(indices)
        
    # Initial solution: Greedy Set Cover
    # We need a fast greedy to start SA.
    # Standard greedy: pick block covering most uncovered.
    # To speed up, we can use a heuristic or just run a few steps.
    # Given the budget, we can afford a decent greedy.
    
    def get_greedy_initial():
        random.seed(42) # Fixed seed for reproducibility of initial, SA handles rest
        uncovered = np.ones(n_t, dtype=np.int32)
        selected_blocks = []
        
        # To make greedy faster, we can sample blocks if n_blocks is large, 
        # but 75k is manageable if we do it smartly.
        # However, iterating 75k blocks * 300 steps * 56 ops = 1.26B ops. Too slow in Python.
        
        # Optimized Greedy:
        # Maintain a list of active blocks.
        # Or, use a different initialization: Random valid cover?
        # Hard to find a valid cover randomly with few blocks.
        
        # Let's use a "Chvatal" style greedy with bitsets for speed?
        # Or, just use a simpler randomized greedy that is fast enough.
        # We can pre-compute the coverage of each block as a bitmask?
        # n_t = 11628 bits. 11628/64 ~ 182 uint64s.
        # 75k blocks * 182 uint64s * 8 bytes ~ 110 MB. Feasible.
        
        # Let's use Python integers as bitmasks.
        # block_mask[i] is an int with bits set for t-subsets covered by block i.
        block_masks = [0] * n_blocks
        for i, indices in enumerate(block_t_indices):
            mask = 0
            for idx in indices:
                mask |= (1 << idx)
            block_masks[i] = mask
            
        full_mask = (1 << n_t) - 1
        current_uncovered = full_mask
        
        # Greedy loop
        # To speed up, we can limit the search space or use a heuristic.
        # Let's try a standard greedy but with a time check.
        # If it takes too long, we fall back to a random large set.
        
        # Actually, for v=19, greedy might be slow.
        # Let's use a "Randomized Greedy" with a limit on iterations or steps.
        # Or, better: Initialize with a random set of blocks that is likely valid?
        # No, let's try to make the greedy fast.
        
        # Optimization: Pre-compute nothing else. Just use bit_count.
        # (uncovered & mask).bit_count() is fast in Python 3.8+.
        
        for step in range(500): # Max steps
            if time.time() > deadline - 60: # Reserve time for SA
                break
                
            best_block = -1
            best_gain = -1
            
            # Sample a subset of blocks to speed up?
            # If we sample 1000 blocks, we might miss the best one.
            # But 75k is large. Let's try iterating all but break early if time is tight?
            # No, just iterate all. 75k * 300 steps = 22.5M iterations.
            # Each iteration: bit_count. ~1us? 22.5s. Too slow.
            
            # We need a faster initial solution.
            # Idea: Use a pre-computed "good" set? No.
            # Idea: Use a heuristic greedy that is approximate.
            # Let's use a "Randomized Greedy" where we pick the best from a random sample of 500 blocks.
            # This is much faster: 300 steps * 500 samples = 150k ops. Very fast.
            # It might produce a worse initial cover, but SA can fix it.
            
            sample_size = 500
            candidates = random.sample(range(n_blocks), sample_size)
            
            for i in candidates:
                gain = (current_uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = i
                elif gain == best_gain and random.random() < 0.1:
                    best_block = i
                    
            if best_block == -1 or best_gain == 0:
                # No improvement found in sample. Try full scan? Or break?
                # If gain is 0, we are stuck. Break.
                if best_gain == 0:
                    break
                # Else, maybe try full scan once?
                # For now, break to save time.
                break
                
            selected_blocks.append(all_blocks[best_block])
            current_uncovered &= ~block_masks[best_block]
            
            if current_uncovered == 0:
                break
                
        return selected_blocks

    # Get initial solution
    initial_blocks = get_greedy_initial()
    
    if not initial_blocks:
        # Fallback: Random blocks?
        # Just return empty or random?
        # We need a valid cover.
        # If greedy failed, we might have an invalid cover.
        # Let's ensure validity by adding random blocks until valid.
        # This is expensive. 
        # For robustness, let's assume greedy works reasonably well for v=19.
        # If not, we return the best we have (might be invalid, score 0).
        # But let's try to fix it.
        
        # Check validity
        uncovered = (1 << n_t) - 1
        for b in initial_blocks:
            idx = block_idx_map[b]
            uncovered &= ~block_masks[idx]
            
        if uncovered != 0:
            # Add random blocks until valid
            while uncovered != 0 and time.time() < deadline - 10:
                b = random.choice(all_blocks)
                if b not in initial_blocks:
                    initial_blocks.append(b)
                    idx = block_idx_map[b]
                    uncovered &= ~block_masks[idx]
                    
    # Convert to indices for SA
    current_solution = [block_idx_map[b] for b in initial_blocks]
    
    # Calculate initial objective
    # Objective: Uncovered count + lambda * Block count
    # We want to minimize this.
    # Uncovered count can be calculated by summing (counts == 0).
    
    # Maintain coverage counts
    counts = np.zeros(n_t, dtype=np.int32)
    for idx in current_solution:
        for t_idx in block_t_indices[idx]:
            counts[t_idx] += 1
            
    uncovered_count = np.sum(counts == 0)
    block_count = len(current_solution)
    
    # Objective function
    # Penalty for uncovered should be high.
    # Let's use: cost = uncovered_count * 1000 + block_count
    
    def get_cost(counts, n_blocks_in_sol):
        unc = np.sum(counts == 0)
        return unc * 1000 + n_blocks_in_sol
        
    current_cost = get_cost(counts, block_count)
    best_cost = current_cost
    best_solution = list(current_solution)
    
    # Simulated Annealing
    # Temperature schedule
    T = 100.0
    cooling_rate = 0.999
    
    # We will perform moves:
    # 1. Add a random block
    # 2. Remove a random block (if valid? No, SA allows invalid)
    # 3. Swap a block with another
    
    # To keep it simple and fast, let's use Add/Remove/Swap on the list of indices.
    # We need to update counts efficiently.
    
    # Pre-compute block_t_indices as numpy arrays for faster iteration?
    # block_t_indices is a list of lists.
    
    max_iters = 20000
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
            
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.3:
            # Add a random block
            if not current_solution:
                continue
            new_block = random.randint(0, n_blocks - 1)
            if new_block in current_solution:
                continue
                
            # Apply
            current_solution.append(new_block)
            for t_idx in block_t_indices[new_block]:
                counts[t_idx] += 1
                
            new_block_count = len(current_solution)
            new_cost = get_cost(counts, new_block_count)
            
            # Acceptance
            delta = new_cost - current_cost
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_solution = list(current_solution)
            else:
                # Revert
                current_solution.pop()
                for t_idx in block_t_indices[new_block]:
                    counts[t_idx] -= 1
                    
        elif move_type < 0.6:
            # Remove a random block
            if not current_solution:
                continue
            rem_idx = random.randint(0, len(current_solution) - 1)
            rem_block = current_solution[rem_idx]
            
            # Apply
            current_solution.pop(rem_idx)
            for t_idx in block_t_indices[rem_block]:
                counts[t_idx] -= 1
                
            new_block_count = len(current_solution)
            new_cost = get_cost(counts, new_block_count)
            
            # Acceptance
            delta = new_cost - current_cost
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_solution = list(current_solution)
            else:
                # Revert
                current_solution.insert(rem_idx, rem_block)
                for t_idx in block_t_indices[rem_block]:
                    counts[t_idx] += 1
                    
        else:
            # Swap a block
            if len(current_solution) < 2:
                continue
            swap_idx = random.randint(0, len(current_solution) - 1)
            old_block = current_solution[swap_idx]
            
            # Pick a new block not in solution
            new_block = random.randint(0, n_blocks - 1)
            if new_block == old_block:
                continue
            # Check if new_block is already in solution (excluding the one we're swapping)
            # This check is O(N). N is ~300. Acceptable.
            if new_block in current_solution:
                # If it's the same index, skip. If different, we are effectively moving it.
                # But we want to replace old_block with new_block.
                # If new_block is already there, we would have duplicates.
                # Let's just skip if new_block is in solution.
                continue
                
            # Apply
            current_solution[swap_idx] = new_block
            for t_idx in block_t_indices[old_block]:
                counts[t_idx] -= 1
            for t_idx in block_t_indices[new_block]:
                counts[t_idx] += 1
                
            new_block_count = len(current_solution)
            new_cost = get_cost(counts, new_block_count)
            
            # Acceptance
            delta = new_cost - current_cost
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_solution = list(current_solution)
            else:
                # Revert
                current_solution[swap_idx] = old_block
                for t_idx in block_t_indices[new_block]:
                    counts[t_idx] -= 1
                for t_idx in block_t_indices[old_block]:
                    counts[t_idx] += 1
                    
        # Cool down
        T *= cooling_rate
        if T < 0.1:
            T = 0.1 # Floor temperature to allow some exploration
            
    # Final Pruning
    # Remove redundant blocks from best_solution
    # We need to check if a block is essential.
    # A block is essential if it is the only one covering some t-subset.
    
    # Rebuild counts for best_solution
    final_counts = np.zeros(n_t, dtype=np.int32)
    for idx in best_solution:
        for t_idx in block_t_indices[idx]:
            final_counts[t_idx] += 1
            
    pruned_solution = []
    for idx in best_solution:
        # Check if essential
        essential = False
        for t_idx in block_t_indices[idx]:
            if final_counts[t_idx] == 1:
                essential = True
                break
        if essential:
            pruned_solution.append(idx)
            # Update counts for remaining blocks?
            # We need to be careful. If we keep it, counts stay.
            # If we remove it, counts decrease.
            # Since we are iterating, we should update counts as we remove.
            # But we only remove if NOT essential.
            # So we don't update counts for kept blocks.
            # We DO update counts for removed blocks.
        else:
            # Remove
            for t_idx in block_t_indices[idx]:
                final_counts[t_idx] -= 1
                
    # Verify validity
    if np.any(final_counts == 0):
        # Fallback to best_solution if pruning broke it (shouldn't happen if logic is right)
        pruned_solution = best_solution
        
    # Convert back to blocks
    final_blocks = [all_blocks[i] for i in pruned_solution]
    
    return final_blocks
