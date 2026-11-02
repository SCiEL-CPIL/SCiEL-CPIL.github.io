import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Pre-compute triple indices
    # Map each tuple (a,b,c) with a<b<c to an index 0..C(v,t)-1
    all_triples = list(itertools.combinations(range(v), t))
    n_triples = len(all_triples)
    triple_to_idx = {trip: i for i, trip in enumerate(all_triples)}
    
    # Function to get indices of triples in a block
    def block_to_triple_indices(block):
        return [triple_to_idx[tuple(sorted(t))] for t in itertools.combinations(block, t)]

    # Greedy Construction
    uncovered = np.ones(n_triples, dtype=np.int8) # 1 if uncovered
    blocks = []
    
    # Pre-compute all possible blocks? 
    # C(24,7) is ~346k. 
    # It might be faster to generate candidates on the fly or pre-compute if memory allows.
    # Let's pre-compute the triple indices for all blocks to speed up greedy and local search.
    # Memory: 346k blocks * 35 triples * 4 bytes (int32) ~ 48 MB. Acceptable.
    
    # However, generating all blocks takes time.
    # Greedy step:
    # 1. Find an uncovered triple.
    # 2. Find all blocks containing it.
    # 3. Pick the one covering most uncovered triples.
    
    # To do this efficiently without pre-computing ALL blocks initially:
    # We can generate blocks containing a specific triple.
    # If triple is (a,b,c), the other k-t points are chosen from the remaining v-t points.
    
    def get_candidate_blocks(triple):
        # triple is a tuple of t points
        remaining_points = [p for p in range(v) if p not in triple]
        # Choose k-t points from remaining
        for comb in itertools.combinations(remaining_points, k - t):
            block = tuple(sorted(triple + comb))
            yield block

    # Greedy Loop
    while np.any(uncovered):
        if time.time() > deadline:
            break
        
        # Pick first uncovered triple
        uncov_indices = np.where(uncovered == 1)[0]
        if len(uncov_indices) == 0:
            break
        target_idx = uncov_indices[0]
        target_triple = all_triples[target_idx]
        
        best_block = None
        best_gain = -1
        
        # Iterate candidates
        for block in get_candidate_blocks(target_triple):
            # Calculate gain: number of uncovered triples covered by this block
            trip_indices = block_to_triple_indices(block)
            # Count how many of these are currently uncovered
            gain = np.sum(uncovered[trip_indices])
            if gain > best_gain:
                best_gain = gain
                best_block = block
        
        if best_block is None:
            # Should not happen if v >= k
            break
            
        # Add block
        blocks.append(best_block)
        trip_indices = block_to_triple_indices(best_block)
        uncovered[trip_indices] = 0
        
    # If we timed out during greedy, we might have an invalid covering.
    # We must ensure validity. If uncovered > 0, we must finish greedy or fix it.
    # The deadline check is inside the loop. If we exit due to time, we need to handle it.
    # Better: Run greedy to completion if it's fast enough. 
    # C(24,7,3) greedy is usually fast (< 1s).
    
    # Local Search: Simulated Annealing
    # State: list of blocks
    # Objective: Minimize len(blocks) subject to all uncovered == 0
    # We will use a penalty approach if we allow invalid states during search,
    # but it's safer to stay in the valid set or use a strict penalty.
    
    # Let's use a strict local search:
    # 1. Try to remove blocks that are redundant.
    # 2. Try to swap blocks to improve "potential" for removal.
    
    # First, normalize blocks to tuples for hashing
    blocks = [tuple(b) for b in blocks]
    block_set = set(blocks)
    
    # Calculate coverage counts
    counts = np.zeros(n_triples, dtype=np.int32)
    for b in blocks:
        idxs = block_to_triple_indices(b)
        counts[idxs] += 1
        
    # Ensure validity
    if np.any(counts == 0):
        # If invalid, we must run greedy to finish
        while np.any(counts == 0):
            uncov_indices = np.where(counts == 0)[0]
            target_idx = uncov_indices[0]
            target_triple = all_triples[target_idx]
            
            best_block = None
            best_gain = -1
            for block in get_candidate_blocks(target_triple):
                if block in block_set:
                    continue
                trip_indices = block_to_triple_indices(block)
                gain = np.sum(counts[trip_indices] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            if best_block:
                blocks.append(best_block)
                block_set.add(best_block)
                idxs = block_to_triple_indices(best_block)
                counts[idxs] += 1
            else:
                # Fallback: just add a random block containing the triple
                # This shouldn't happen
                break

    # Local Search Loop
    T = 1.0
    best_blocks = list(blocks)
    best_len = len(blocks)
    
    while time.time() < deadline:
        # 1. Try to remove redundant blocks
        removed = False
        for i in range(len(blocks) - 1, -1, -1):
            b = blocks[i]
            idxs = block_to_triple_indices(b)
            # Check if any triple has count 1
            if np.any(counts[idxs] == 1):
                continue
            # Safe to remove
            counts[idxs] -= 1
            blocks.pop(i)
            block_set.discard(b)
            removed = True
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = list(blocks)
        
        if removed:
            T = 1.0 # Reset temperature on improvement
            continue
            
        # 2. Local Search Move: Swap
        # Pick a block to remove
        i = random.randint(0, len(blocks) - 1)
        b_remove = blocks[i]
        idxs_remove = block_to_triple_indices(b_remove)
        
        # Simulate removal
        counts_sim = counts.copy()
        counts_sim[idxs_remove] -= 1
        
        # Identify uncovered triples
        uncov_mask = (counts_sim == 0)
        uncov_count = np.sum(uncov_mask)
        
        if uncov_count == 0:
            # This block was redundant? We should have caught it above.
            # Maybe counts were updated?
            continue

        # Find a block that covers the most uncovered triples
        # We can search among blocks containing the most "critical" uncovered triple
        # Or just random search?
        # Better: Pick the uncovered triple with the fewest candidates?
        # Or pick a random uncovered triple and find a block covering it.
        
        uncov_indices = np.where(uncov_mask)[0]
        # Pick a random uncovered triple to anchor the candidate search
        target_idx = uncov_indices[random.randint(0, len(uncov_indices) - 1)]
        target_triple = all_triples[target_idx]
        
        best_swap_block = None
        best_uncov_after_swap = uncov_count
        
        # Sample candidates
        # Instead of all, sample some
        candidates = list(get_candidate_blocks(target_triple))
        random.shuffle(candidates)
        # Limit candidates for speed
        for b_add in candidates[:50]: 
            if b_add == b_remove:
                continue
            idxs_add = block_to_triple_indices(b_add)
            # Calculate uncovered after adding b_add
            # new_counts = counts_sim + indicator(b_add)
            # We want to minimize number of zeros in new_counts
            # This is equivalent to maximizing number of 1s in counts_sim that are covered by b_add
            # But we also care about triples that are 0 in counts_sim but covered by b_add.
            
            # Let's just compute the new uncovered count
            # Efficient check:
            # A triple is uncovered in final state if:
            # (counts_sim[t] == 0 AND t NOT in b_add) OR (counts_sim[t] > 0 AND t NOT in b_add ... wait)
            # Actually:
            # Final count[t] = counts_sim[t] + (1 if t in b_add else 0)
            # Uncovered if Final count[t] == 0
            # This happens if counts_sim[t] == 0 AND t NOT in b_add
            
            # So, new_uncovered_count = (uncov_count - (number of uncov triples in b_add))
            # We want to maximize (number of uncov triples in b_add)
            
            # Count how many of the currently uncovered triples are in b_add
            # uncov_mask is boolean array
            # We need to check intersection of idxs_add and uncov_indices
            
            # Vectorized:
            # Create a set of idxs_add for fast lookup? No, numpy boolean indexing is faster.
            # But idxs_add is a list of indices.
            # Let's create a boolean array for b_add? No, that's O(n_triples).
            
            # Simpler:
            # sum(1 for idx in idxs_add if uncov_mask[idx])
            gain = np.sum(uncov_mask[idxs_add])
            
            new_uncov = uncov_count - gain
            
            if new_uncov < best_uncov_after_swap:
                best_uncov_after_swap = new_uncov
                best_swap_block = b_add
            elif new_uncov == best_uncov_after_swap and best_swap_block is None:
                best_swap_block = b_add
                
        if best_swap_block is not None:
            # Calculate energy change
            # Old state: valid (uncov_count from original counts was 0? No, we are in local search, original is valid)
            # Wait, the original state is valid (counts > 0 for all).
            # We removed b_remove, creating uncov_count > 0.
            # We add best_swap_block.
            # If best_uncov_after_swap == 0, the new state is valid.
            # If best_uncov_after_swap > 0, the new state is invalid.
            
            # We should only accept if valid, or use SA.
            # For now, accept only if valid.
            if best_uncov_after_swap == 0:
                # Apply swap
                counts[idxs_remove] -= 1
                blocks[i] = best_swap_block
                block_set.discard(b_remove)
                block_set.add(best_swap_block)
                idxs_add = block_to_triple_indices(best_swap_block)
                counts[idxs_add] += 1
                
                # Check for improvements (redundant blocks)
                # Loop back to check removals
                
                if len(blocks) < best_len:
                    best_len = len(blocks)
                    best_blocks = list(blocks)
            else:
                # Invalid, reject
                pass
        else:
            # No good swap found
            pass
            
        # Cooling
        T *= 0.99
        
    # Return the best valid covering found
    # Ensure best_blocks is valid
    # It should be, as we only update best_blocks when we have a valid state (after removals)
    # But what if the initial greedy was invalid and we never fixed it?
    # The "Ensure validity" block above fixes it.
    
    return [list(b) for b in best_blocks]
