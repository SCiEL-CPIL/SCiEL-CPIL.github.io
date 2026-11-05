import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Precompute t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    n_tsubs = len(all_tsubsets)
    
    # Precompute all blocks and their covered t-subset indices
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map block tuple to index in all_blocks
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    # Precompute covered indices for each block
    # block_cover[i] is a list of t-subset indices covered by block i
    block_cover = [None] * n_blocks
    for i, b in enumerate(all_blocks):
        block_cover[i] = [tsub_to_idx[ts] for ts in itertools.combinations(b, t)]
        
    if time.time() > deadline - 5:
        return [list(b) for b in all_blocks[:100]] # Fallback, likely invalid but fast

    # Phase 1: Greedy/UCB Initialization
    # We will run a few variations of greedy to get a good starting point.
    # Variation 1: Standard Greedy (Max Coverage)
    # Variation 2: UCB-based Greedy
    
    best_initial = None
    best_initial_len = float('inf')
    
    # Standard Greedy
    uncovered = [1] * n_tsubs
    n_uncovered = n_tsubs
    selected = []
    
    # For efficiency in standard greedy, we can just iterate. 
    # But with 346k blocks, iterating all each step is slow if many steps.
    # However, greedy usually converges in < 100 steps for this size.
    # Let's use a simple loop.
    
    # Optimization: Maintain a list of uncovered indices to sample? 
    # No, we need max gain.
    # We can precompute for each t-subset, the blocks covering it? 
    # That's the inverse map.
    # tsub_to_blocks = defaultdict(list)
    # for b_idx, covers in enumerate(block_cover):
    #     for ts_idx in covers:
    #         tsub_to_blocks[ts_idx].append(b_idx)
    
    # This inverse map is useful for UCB and for quick lookup.
    tsub_to_blocks = defaultdict(list)
    for b_idx, covers in enumerate(block_cover):
        for ts_idx in covers:
            tsub_to_blocks[ts_idx].append(b_idx)
            
    # Standard Greedy
    selected_greedy = []
    uncovered = [1] * n_tsubs
    n_uncovered = n_tsubs
    
    while n_uncovered > 0 and time.time() < deadline - 30:
        best_gain = -1
        best_block_idx = -1
        
        # Iterate through all blocks? 346k * ~80 steps = 27M checks. 
        # Each check is sum of 35 bools. 
        # In Python this might be slow.
        # Let's try to optimize.
        # We can maintain a priority queue? Hard to update.
        # Let's just do it. If it's too slow, we'll see.
        
        # Actually, a smarter greedy:
        # Pick a random uncovered t-subset.
        # Look at all blocks covering it.
        # Pick the one with max gain among those.
        # This reduces candidate blocks from 346k to ~C(21,4)=5985 (blocks containing a specific triple in 24 points).
        # Wait, a triple (3 points) is contained in C(24-3, 7-3) = C(21, 4) = 5985 blocks.
        # This is much faster.
        
        # Find an uncovered t-subset
        # We can keep a list of uncovered indices?
        # Or just pick random until we find one.
        # Or maintain a set.
        
        # Let's maintain a list of uncovered indices for fast random access?
        # Or just a boolean array and a count.
        # To pick a random uncovered:
        #   while True:
        #       idx = random.randint(0, n_tsubs-1)
        #       if uncovered[idx]: break
        
        if n_uncovered == 0:
            break
            
        # Find a random uncovered t-subset
        while True:
            ts_idx = random.randint(0, n_tsubs - 1)
            if uncovered[ts_idx]:
                break
        
        # Get blocks covering this t-subset
        candidates = tsub_to_blocks[ts_idx]
        
        best_gain = -1
        best_block_idx = -1
        
        for b_idx in candidates:
            gain = 0
            for ts in block_cover[b_idx]:
                if uncovered[ts]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
                
        if best_block_idx == -1:
            break
            
        # Select block
        selected_greedy.append(best_block_idx)
        
        # Update uncovered
        for ts in block_cover[best_block_idx]:
            if uncovered[ts]:
                uncovered[ts] = 0
                n_uncovered -= 1
                
    if n_uncovered == 0 and len(selected_greedy) < best_initial_len:
        best_initial = selected_greedy
        best_initial_len = len(selected_greedy)

    # Phase 2: Local Search (Simulated Annealing)
    # Start from best_initial
    if best_initial is None:
        # Fallback if greedy failed (shouldn't happen)
        return []
        
    current_blocks = list(best_initial)
    
    # Calculate cover counts
    cover_count = np.zeros(n_tsubs, dtype=np.int32)
    for b_idx in current_blocks:
        for ts_idx in block_cover[b_idx]:
            cover_count[ts_idx] += 1
            
    n_uncovered = np.sum(cover_count == 0)
    
    best_blocks = list(current_blocks)
    best_len = len(current_blocks)
    
    # SA Parameters
    T = 10.0
    cooling = 0.995
    min_T = 0.01
    
    rng = random.Random(123)
    
    while T > min_T and time.time() < deadline - 5:
        move_type = rng.random()
        
        if move_type < 0.4:
            # Remove a block
            if len(current_blocks) == 0:
                continue
            idx = rng.randrange(len(current_blocks))
            b_idx = current_blocks[idx]
            
            # Check if safe
            safe = True
            for ts_idx in block_cover[b_idx]:
                if cover_count[ts_idx] == 1:
                    safe = False
                    break
            
            if safe:
                # Accept
                for ts_idx in block_cover[b_idx]:
                    cover_count[ts_idx] -= 1
                current_blocks.pop(idx)
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = list(current_blocks)
            else:
                # Calculate delta
                # delta_uncovered = number of ts with count 1
                delta_unc = 0
                for ts_idx in block_cover[b_idx]:
                    if cover_count[ts_idx] == 1:
                        delta_unc += 1
                delta_cost = -1 + 1000 * delta_unc
                if delta_cost < 0 or rng.random() < np.exp(-delta_cost / T):
                    for ts_idx in block_cover[b_idx]:
                        cover_count[ts_idx] -= 1
                    current_blocks.pop(idx)
                    n_uncovered += delta_unc
                    if len(current_blocks) < best_len and n_uncovered == 0:
                        best_len = len(current_blocks)
                        best_blocks = list(current_blocks)
                        
        elif move_type < 0.8:
            # Add a block
            if n_uncovered == 0:
                continue # Don't add if already valid, focus on removal
                
            # Pick a random uncovered t-subset
            found = False
            for _ in range(20):
                ts_idx = rng.randrange(n_tsubs)
                if cover_count[ts_idx] == 0:
                    found = True
                    break
            if not found:
                continue
                
            candidates = tsub_to_blocks[ts_idx]
            if not candidates:
                continue
                
            b_idx = rng.choice(candidates)
            
            if b_idx in current_blocks:
                continue
                
            # Calculate gain
            gain = 0
            for ts_idx in block_cover[b_idx]:
                if cover_count[ts_idx] == 0:
                    gain += 1
                    
            delta_cost = 1 - 1000 * gain
            if delta_cost < 0 or rng.random() < np.exp(-delta_cost / T):
                for ts_idx in block_cover[b_idx]:
                    cover_count[ts_idx] += 1
                current_blocks.append(b_idx)
                n_uncovered -= gain
                if len(current_blocks) < best_len and n_uncovered == 0:
                    best_len = len(current_blocks)
                    best_blocks = list(current_blocks)
                    
        else:
            # Swap
            if len(current_blocks) == 0:
                continue
            idx_old = rng.randrange(len(current_blocks))
            old_b_idx = current_blocks[idx_old]
            
            # Pick new block
            # Heuristic: Pick from blocks covering an uncovered t-subset if any, else random
            if n_uncovered > 0:
                found = False
                for _ in range(20):
                    ts_idx = rng.randrange(n_tsubs)
                    if cover_count[ts_idx] == 0:
                        found = True
                        break
                if found:
                    candidates = tsub_to_blocks[ts_idx]
                    if candidates:
                        new_b_idx = rng.choice(candidates)
                    else:
                        new_b_idx = rng.randrange(n_blocks)
                else:
                    new_b_idx = rng.randrange(n_blocks)
            else:
                new_b_idx = rng.randrange(n_blocks)
                
            if new_b_idx == old_b_idx or new_b_idx in current_blocks:
                continue
                
            # Apply swap
            # Remove old
            for ts_idx in block_cover[old_b_idx]:
                cover_count[ts_idx] -= 1
            # Add new
            for ts_idx in block_cover[new_b_idx]:
                cover_count[ts_idx] += 1
                
            # Check validity and cost
            # We need to know if n_uncovered changed.
            # Since we don't track n_uncovered efficiently in swap without recounting or careful delta,
            # and swap keeps n_blocks same, we only accept if it improves validity (reduces n_uncovered).
            # If n_uncovered was 0, we only accept if it stays 0 (which it might not).
            # Actually, if n_uncovered was 0, swap might break it.
            # If n_uncovered > 0, swap might fix it.
            
            # Simple check: Recount uncovered? Too slow.
            # Let's just accept swaps that don't increase uncovered count significantly?
            # Or just skip swaps for now to keep it robust?
            # Let's try to calculate delta_uncovered properly.
            
            # Revert for now if we can't calculate quickly.
            # Let's implement a fast delta check for swap.
            
            # We need to know which t-subsets are affected.
            # affected = set(block_cover[old_b_idx]) | set(block_cover[new_b_idx])
            # For each ts in affected:
            #   old_count = cover_count[ts] (after swap)
            #   We need old_old_count (before swap).
            #   before_swap_count[ts] = cover_count[ts] + (1 if ts in old else 0) - (1 if ts in new else 0)
            
            # This is getting complex. Let's just revert and skip swap for simplicity/safety.
            for ts_idx in block_cover[new_b_idx]:
                cover_count[ts_idx] -= 1
            for ts_idx in block_cover[old_b_idx]:
                cover_count[ts_idx] += 1
                
            # Actually, let's try a simple swap acceptance:
            # Accept if it reduces n_uncovered.
            # We can estimate gain/loss.
            # Gain: ts in new, not in old, with count 0 (before swap) -> becomes 1.
            # Loss: ts in old, not in new, with count 1 (before swap) -> becomes 0.
            
            gain = 0
            loss = 0
            for ts_idx in block_cover[new_b_idx]:
                if ts_idx not in block_cover[old_b_idx]: # Optimization: check membership
                    if cover_count[ts_idx] == 0:
                        gain += 1
            for ts_idx in block_cover[old_b_idx]:
                if ts_idx not in block_cover[new_b_idx]:
                    if cover_count[ts_idx] == 1:
                        loss += 1
                        
            delta_unc = loss - gain
            if delta_unc < 0 or (delta_unc == 0 and rng.random() < 0.1): # Accept if improves or neutral sometimes
                # Apply
                for ts_idx in block_cover[old_b_idx]:
                    cover_count[ts_idx] -= 1
                for ts_idx in block_cover[new_b_idx]:
                    cover_count[ts_idx] += 1
                current_blocks[idx_old] = new_b_idx
                n_uncovered += delta_unc
                if len(current_blocks) < best_len and n_uncovered == 0:
                    best_len = len(current_blocks)
                    best_blocks = list(current_blocks)
                elif n_uncovered < 0: # Should not happen
                    n_uncovered = 0

        T *= cooling
        
        # Periodically check if we have a valid solution
        # n_uncovered is tracked, so we can check it.
        
    # Final Polish: Remove redundant blocks
    if time.time() < deadline - 2:
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            for i in range(len(current_blocks) - 1, -1, -1):
                b_idx = current_blocks[i]
                redundant = True
                for ts_idx in block_cover[b_idx]:
                    if cover_count[ts_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    for ts_idx in block_cover[b_idx]:
                        cover_count[ts_idx] -= 1
                    current_blocks.pop(i)
                    changed = True
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = list(current_blocks)

    return [list(all_blocks[b]) for b in best_blocks]
