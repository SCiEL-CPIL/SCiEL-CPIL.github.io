import itertools
import numpy as np
import time
import random
import heapq
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Precomputation
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    all_tsubsets = list(itertools.combinations(points, t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Bitsets for fast set operations
    # block_masks[b] is an integer where bit i is 1 if block b covers tsubset i
    block_masks = [0] * n_blocks
    # tsub_blocks[i] is a list of block indices that cover tsubset i
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    
    # Precompute masks
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        # Generate all t-subsets of this block
        for ts in itertools.combinations(block, t):
            idx = tsub_idx[ts]
            mask |= (1 << idx)
            tsub_blocks[idx].append(b_idx)
        block_masks[b_idx] = mask
        
    full_mask = (1 << n_tsubsets) - 1
    
    # Helper to check if a block covers all tsubs in a set (represented as list of indices)
    # We will use bit operations for this in the main loop for speed
    
    # 2. Greedy Initialization
    # We can try multiple restarts with different weightings, but a good greedy is usually close.
    # Let's use a weighted greedy where weights are initially 1, but we can perturb them.
    # For simplicity and speed, we'll run a few greedy restarts with slight perturbations.
    
    best_solution = None
    best_size = float('inf')
    
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
            
        # Initialize weights
        # For restart 0, uniform. For others, add small noise to break ties and explore.
        weights = np.ones(n_blocks)
        if restart > 0:
            # Perturb weights
            noise = np.random.uniform(0, 0.1, n_blocks)
            weights += noise
            
        # Greedy selection
        selected_blocks = []
        selected_set = set()
        uncovered_mask = full_mask
        gains = np.zeros(n_blocks)
        
        # Initial gains: number of tsubs covered by each block
        for b_idx in range(n_blocks):
            gains[b_idx] = block_masks[b_idx].bit_count()
            
        # Heap of (-gain, block_idx)
        heap = [(-gains[b_idx], b_idx) for b_idx in range(n_blocks)]
        heapq.heapify(heap)
        
        while uncovered_mask:
            if time.time() > deadline - 25:
                break
                
            best_block = -1
            best_gain = 0
            valid_gain = 0
            
            # Pop stale entries
            while heap:
                neg_g, b_idx = heapq.heappop(heap)
                g = -neg_g
                # Check if this entry is still valid
                # A block is invalid if it's already selected
                if b_idx in selected_set:
                    continue
                # The gain stored in heap might be stale if the block's effective gain changed
                # However, gains only decrease. So if heap_gain > current_gain, it's stale.
                # But we don't store current_gain separately in a fast-access array that updates easily?
                # Actually, we can re-calculate gain if needed, or just trust the heap if we push updates.
                # Standard approach: push new gain when it changes.
                # Here, we push initial gains. As we cover tsubs, gains decrease.
                # We need to know the *current* gain to compare.
                # Let's maintain a `current_gain` array.
                pass
            
            # The above heap logic is tricky with stale entries without a current_gain array.
            # Let's use a simpler approach for the greedy phase given the small N (50k blocks).
            # Or, maintain `current_gain` array and push to heap whenever it changes.
            
            # Reset for a cleaner greedy implementation using current_gain array
            if not selected_blocks:
                 current_gains = np.array([block_masks[b].bit_count() for b in range(n_blocks)], dtype=np.int32)
                 # Apply weights? The problem is minimizing count, so uniform weight is standard.
                 # Perturbation helps in ties.
                 effective_gains = current_gains * weights
                 heap = [(-effective_gains[b], b) for b in range(n_blocks)]
                 heapq.heapify(heap)
            
            # This restart loop is getting complicated. Let's simplify:
            # Run one high-quality greedy, then do extensive local search.
            # The local search is where the improvement comes from.
            if restart > 0:
                continue # Skip extra restarts if we focus on local search quality
                
            # Single Greedy Run
            current_gains = np.array([block_masks[b].bit_count() for b in range(n_blocks)], dtype=np.int32)
            # Add small random noise to break ties deterministically but diversely?
            # No, let's just use the gains.
            heap = [(-current_gains[b], b) for b in range(n_blocks)]
            heapq.heapify(heap)
            
            while uncovered_mask:
                if time.time() > deadline - 25:
                    break
                    
                best_block = -1
                best_gain = 0
                
                while heap:
                    neg_g, b_idx = heapq.heappop(heap)
                    g = -neg_g
                    if b_idx in selected_set:
                        continue
                    # Check if gain is still valid. 
                    # Since gains only decrease, if g > current_gains[b_idx], it's stale.
                    if g > current_gains[b_idx]:
                        continue
                    # If g == current_gains[b_idx], it's valid.
                    # Note: It's possible current_gains[b_idx] is lower than g, meaning it's stale.
                    # If current_gains[b_idx] is higher? No, gains don't increase.
                    best_block = b_idx
                    best_gain = g
                    break
                
                if best_block == -1:
                    break
                    
                selected_blocks.append(best_block)
                selected_set.add(best_block)
                
                # Update uncovered mask
                new_covered = block_masks[best_block] & uncovered_mask
                uncovered_mask &= ~block_masks[best_block]
                
                # Update gains for all blocks that cover any of the newly covered tsubs
                # Iterate through bits in new_covered
                m = new_covered
                while m:
                    lsb = m & -m
                    ts_idx = lsb.bit_length() - 1
                    # For each block covering ts_idx, decrease gain
                    for b_prime in tsub_blocks[ts_idx]:
                        if b_prime in selected_set:
                            continue
                        current_gains[b_prime] -= 1
                        heapq.heappush(heap, (-current_gains[b_prime], b_prime))
                    m ^= lsb
                    
        if not selected_blocks:
            continue
            
        # 3. Local Search
        # Convert selected_blocks to a set for O(1) lookup
        sol_set = set(selected_blocks)
        
        # Compute cover counts for each tsubset
        cover_counts = np.zeros(n_tsubsets, dtype=np.int32)
        for b in sol_set:
            # Iterate through tsubs in block b
            mask = block_masks[b]
            while mask:
                lsb = mask & -mask
                ts_idx = lsb.bit_length() - 1
                cover_counts[ts_idx] += 1
                mask ^= lsb
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Phase 1: Deletion
            # Check if any block can be removed
            to_remove = []
            for b in sol_set:
                redundant = True
                mask = block_masks[b]
                while mask:
                    lsb = mask & -mask
                    ts_idx = lsb.bit_length() - 1
                    if cover_counts[ts_idx] == 1:
                        redundant = False
                        break
                    mask ^= lsb
                if redundant:
                    to_remove.append(b)
            
            if to_remove:
                for b in to_remove:
                    sol_set.remove(b)
                    mask = block_masks[b]
                    while mask:
                        lsb = mask & -mask
                        ts_idx = lsb.bit_length() - 1
                        cover_counts[ts_idx] -= 1
                        mask ^= lsb
                improved = True
                continue
            
            # Phase 2: 1-for-1 Swaps
            # For each block in solution, see if we can replace it with another block
            # that covers all its unique tsubs.
            current_sol_list = list(sol_set)
            random.shuffle(current_sol_list)
            
            for b in current_sol_list:
                if time.time() > deadline - 5:
                    break
                if b not in sol_set:
                    continue
                
                # Find unique tsubs covered by b
                unique_tsubs = []
                mask = block_masks[b]
                while mask:
                    lsb = mask & -mask
                    ts_idx = lsb.bit_length() - 1
                    if cover_counts[ts_idx] == 1:
                        unique_tsubs.append(ts_idx)
                    mask ^= lsb
                
                if not unique_tsubs:
                    # Should have been removed in Phase 1, but just in case
                    sol_set.remove(b)
                    mask = block_masks[b]
                    while mask:
                        lsb = mask & -mask
                        ts_idx = lsb.bit_length() - 1
                        cover_counts[ts_idx] -= 1
                        mask ^= lsb
                    improved = True
                    break
                
                # Find a candidate block that covers all unique_tsubs
                # Strategy: Pick the most constrained unique tsub (fewest covering blocks)
                # to minimize the candidate list.
                min_ts = min(unique_tsubs, key=lambda x: len(tsub_blocks[x]))
                candidates = tsub_blocks[min_ts]
                
                # Filter candidates: must cover all other unique_tsubs
                # And must not be in sol_set (unless it's b, which we exclude)
                valid_candidate = -1
                # Sort candidates by some heuristic? e.g., number of unique tsubs it covers?
                # Or just iterate.
                for c in candidates:
                    if c == b:
                        continue
                    if c in sol_set:
                        continue
                    
                    # Check if c covers all unique_tsubs
                    # We can use bit operations:
                    # Does block_masks[c] contain all bits in unique_tsubs_mask?
                    # Let's build unique_tsubs_mask
                    # Actually, we can just check if all ts in unique_tsubs are in block_masks[c]
                    # Since unique_tsubs is small, this is fast.
                    c_mask = block_masks[c]
                    covers_all = True
                    for ts in unique_tsubs:
                        if not (c_mask & (1 << ts)):
                            covers_all = False
                            break
                    if covers_all:
                        valid_candidate = c
                        break
                
                if valid_candidate != -1:
                    # Perform swap
                    # Remove b
                    sol_set.remove(b)
                    mask_b = block_masks[b]
                    while mask_b:
                        lsb = mask_b & -mask_b
                        ts_idx = lsb.bit_length() - 1
                        cover_counts[ts_idx] -= 1
                        mask_b ^= lsb
                        
                    # Add valid_candidate
                    sol_set.add(valid_candidate)
                    mask_c = block_masks[valid_candidate]
                    while mask_c:
                        lsb = mask_c & -mask_c
                        ts_idx = lsb.bit_length() - 1
                        cover_counts[ts_idx] += 1
                        mask_c ^= lsb
                        
                    improved = True
                    break # Restart loop to re-evaluate deletions/swaps
            
        # Update best
        if len(sol_set) < best_size:
            best_size = len(sol_set)
            best_solution = list(sol_set)
            
    # Fallback if no solution found (should not happen)
    if best_solution is None:
        # Simple greedy without local search
        uncovered_mask = full_mask
        res_blocks = []
        while uncovered_mask:
            best_b = -1
            max_cov = -1
            for b in range(n_blocks):
                cov = (block_masks[b] & uncovered_mask).bit_count()
                if cov > max_cov:
                    max_cov = cov
                    best_b = b
            if best_b == -1:
                break
            res_blocks.append(best_b)
            uncovered_mask &= ~block_masks[best_b]
        best_solution = res_blocks
        best_size = len(best_solution)
        
    return [list(all_blocks[idx]) for idx in best_solution]
