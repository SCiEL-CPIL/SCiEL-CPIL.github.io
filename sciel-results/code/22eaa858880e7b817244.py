import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    # Parameters for the specific task C(19, 8, 5)
    # v=19, k=8, t=5
    
    all_points = list(range(v))
    
    # Pre-compute all t-subset masks and map them to indices
    # This allows us to use integer indices for subsets instead of large integers in some structures
    # But integer masks are good for bitwise operations.
    # Let's stick to integer masks for subsets, but map them to 0..N-1 indices for array indexing.
    
    t_subsets = list(itertools.combinations(all_points, t))
    num_t = len(t_subsets)
    
    # Map mask to index
    mask_to_idx = {}
    idx_to_mask = [0] * num_t
    for i, combo in enumerate(t_subsets):
        m = 0
        for p in combo:
            m |= (1 << p)
        mask_to_idx[m] = i
        idx_to_mask[i] = m
        
    # Pre-compute all k-blocks
    k_blocks = list(itertools.combinations(all_points, k))
    num_k = len(k_blocks)
    
    # For each block, compute the list of t-subset indices it covers
    # block_subs[i] = list of subset indices covered by block i
    block_subs = []
    for combo in k_blocks:
        subs = []
        # Generate all t-subsets of this k-block
        for sub in itertools.combinations(combo, t):
            m = 0
            for p in sub:
                m |= (1 << p)
            subs.append(mask_to_idx[m])
        block_subs.append(subs)
        
    # Pre-compute sub_to_blocks: for each t-subset index, list of block indices covering it
    sub_to_blocks = [[] for _ in range(num_t)]
    for b_idx, subs in enumerate(block_subs):
        for s_idx in subs:
            sub_to_blocks[s_idx].append(b_idx)

    # --- Greedy Initialization ---
    def greedy_init():
        cover_count = np.zeros(num_t, dtype=np.int32)
        current_blocks = []
        uncovered = set(range(num_t))
        
        while uncovered:
            # Pick a random uncovered subset to break symmetry
            s_idx = random.choice(list(uncovered))
            
            # Find the best block covering s_idx
            best_block = -1
            best_gain = -1
            
            # Only consider blocks that cover s_idx
            for b_idx in sub_to_blocks[s_idx]:
                # Calculate gain: number of uncovered subsets covered by b_idx
                gain = 0
                for sub_idx in block_subs[b_idx]:
                    if cover_count[sub_idx] == 0:
                        gain += 1
                        if gain > best_gain:
                            # Early exit if we hit max possible? Max is len(block_subs[b_idx])
                            # But we don't know max without iterating all.
                            # C(8,5)=56 is small.
                            pass
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block == -1:
                break # Should not happen
            
            # Add block
            current_blocks.append(best_block)
            for sub_idx in block_subs[best_block]:
                cover_count[sub_idx] += 1
                if cover_count[sub_idx] == 1:
                    uncovered.discard(sub_idx)
                    
        return current_blocks, cover_count

    # --- Local Search: Prune and Swap ---
    def prune_and_swap(current_blocks, cover_count, time_limit):
        # Convert to set for fast membership check
        block_set = set(current_blocks)
        
        improved = True
        start_time = time.time()
        
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            
            # Phase 1: Prune redundant blocks
            # A block is redundant if all its subsets have cover_count > 1
            # We iterate and remove.
            i = 0
            while i < len(current_blocks):
                b_idx = current_blocks[i]
                redundant = True
                for s_idx in block_subs[b_idx]:
                    if cover_count[s_idx] == 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove block
                    # Update cover counts
                    for s_idx in block_subs[b_idx]:
                        cover_count[s_idx] -= 1
                    block_set.remove(b_idx)
                    current_blocks.pop(i)
                    improved = True
                    # Do not increment i, check next element at same index
                else:
                    i += 1
            
            # Phase 2: Swap
            # Try to swap a block to enable further pruning or direct reduction
            # We iterate through blocks. If we swap, we restart the loop.
            swapped = False
            for i in range(len(current_blocks)):
                if (time.time() - start_time) > time_limit:
                    break
                
                b_old = current_blocks[i]
                
                # Temporarily remove b_old
                for s_idx in block_subs[b_old]:
                    cover_count[s_idx] -= 1
                
                # Find uncovered subsets
                uncovered_subs = [s_idx for s_idx in block_subs[b_old] if cover_count[s_idx] == 0]
                
                if not uncovered_subs:
                    # It was redundant, should have been pruned. 
                    # Restore and continue (or just remove if we missed it)
                    for s_idx in block_subs[b_old]:
                        cover_count[s_idx] += 1
                    # Actually, if it's redundant now, prune phase should have caught it.
                    # If we are here, it means it became redundant *after* previous swaps but wasn't pruned?
                    # Prune phase runs at start of while loop. So this shouldn't happen unless we modified cover_count in a way that made it redundant without running prune.
                    # But we are in Swap phase.
                    # Let's just restore and continue.
                    continue
                
                # We need a block b_new that covers all subsets in uncovered_subs
                # Pick the subset with fewest candidates to minimize search
                s_target = min(uncovered_subs, key=lambda x: len(sub_to_blocks[x]))
                
                found_swap = False
                for b_new in sub_to_blocks[s_target]:
                    if b_new in block_set:
                        continue
                    
                    # Check if b_new covers all uncovered_subs
                    valid = True
                    for s_idx in uncovered_subs:
                        if cover_count[s_idx] == 0:
                            # Check if b_new covers s_idx
                            # We can check if s_idx is in block_subs[b_new]
                            # Since block_subs[b_new] is a list, this is O(56).
                            if s_idx not in block_subs[b_new]:
                                valid = False
                                break
                    
                    if valid:
                        # Perform swap
                        # Add b_new
                        for s_idx in block_subs[b_new]:
                            cover_count[s_idx] += 1
                        
                        # Remove b_old (already decremented)
                        block_set.discard(b_old)
                        block_set.add(b_new)
                        
                        # Update current_blocks
                        current_blocks[i] = b_new
                        
                        found_swap = True
                        swapped = True
                        improved = True
                        break
                
                # Restore cover counts for b_old if we didn't swap
                if not found_swap:
                    for s_idx in block_subs[b_old]:
                        cover_count[s_idx] += 1
                
                if swapped:
                    # Restart the while loop to prune again
                    break
        
        return current_blocks, cover_count

    # Main execution
    start_time = time.time()
    time_limit = 110 # Leave buffer
    
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple instances of Greedy + Local Search
    while time.time() - start_time < time_limit:
        # 1. Greedy Init
        blocks, cover_count = greedy_init()
        
        # 2. Local Search (Prune + Swap)
        # Allocate remaining time to this instance
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 5:
            break
            
        blocks, cover_count = prune_and_swap(blocks, cover_count, remaining_time - 1)
        
        # Check if better
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        # If we found a very good solution, we might want to run more local search on it?
        # For now, we start fresh.
        
    # Convert block indices back to lists of points
    if best_blocks is None:
        # Fallback: return empty or raise? Task says valid result required.
        # Greedy should always produce a valid covering.
        # If best_blocks is None, something went wrong.
        # Let's ensure we at least return a valid covering from the last run if best is None.
        # Actually, greedy_init always returns a valid covering.
        # So best_blocks should be set.
        pass
        
    result = [list(k_blocks[b_idx]) for b_idx in best_blocks]
    return result
