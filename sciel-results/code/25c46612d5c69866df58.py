import itertools
import random
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # 1. Precompute all t-sets
    t_sets = list(itertools.combinations(range(v), t))
    t_set_index = {ts: i for i, ts in enumerate(t_sets)}
    num_t_sets = len(t_sets)
    
    # 2. Precompute all k-sets (blocks) and the t-sets they cover
    # This might be memory intensive for larger v, but v=24 is small.
    # C(24,7) = 346104 blocks.
    # Each block covers C(7,3)=35 t-sets.
    # Total entries: 346104 * 35 ~ 12 million.
    # We store for each block index, the list of t-set indices it covers.
    
    # To save memory and time, we can generate blocks on the fly in the greedy step?
    # No, precomputing is better for the "find blocks containing a specific t-set" step.
    
    # Let's map each t-set to the list of blocks that contain it.
    # t_set_to_blocks = {t_set_index: [block_indices]}
    
    # First, generate all blocks and assign them indices.
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Map block to its covered t-set indices
    block_to_tsets = []
    # Map t-set to list of blocks containing it
    tset_to_blocks = [[] for _ in range(num_t_sets)]
    
    for b_idx, block in enumerate(all_blocks):
        # Find t-sets in this block
        tsets_in_block = []
        for ts in itertools.combinations(block, t):
            tset_to_blocks[t_set_index[ts]].append(b_idx)
            tsets_in_block.append(t_set_index[ts])
        block_to_tsets.append(tsets_in_block)
        
    # 3. Greedy Construction with Local Search
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple greedy restarts
    num_restarts = 20 # Adjust based on time
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Greedy Phase
        # uncovered: boolean array, True if not yet covered
        uncovered = [True] * num_t_sets
        num_uncovered = num_t_sets
        current_blocks = []
        
        while num_uncovered > 0:
            # Pick a random uncovered t-set to break ties and vary the path
            # Or pick the first one. Randomization helps escape local minima.
            # Let's pick a random uncovered t-set.
            # To do this efficiently, maintain a list of uncovered indices?
            # Or just loop through until we find one? 
            # Better: maintain a list of uncovered t-set indices.
            
            # For simplicity and speed, let's just pick the first uncovered one found in a random scan?
            # Or, better: pick a random index, if uncovered, use it.
            
            # Optimization: Maintain a list of uncovered t-set indices.
            # When we cover some, we don't remove them from the list immediately, 
            # but we check if they are still uncovered when we pick.
            # This is O(1) pick, O(1) check.
            
            # Let's use a list `uncovered_list` of indices.
            # Actually, the standard greedy picks the t-set with the fewest candidate blocks?
            # No, standard is: pick ANY uncovered t-set, then pick the block covering the most OTHER uncovered t-sets.
            
            # Let's pick a random uncovered t-set.
            while True:
                idx = random.randint(0, num_t_sets - 1)
                if uncovered[idx]:
                    break
            
            # Get candidate blocks for this t-set
            candidates = tset_to_blocks[idx]
            
            best_gain = -1
            best_block_idx = -1
            
            for b_idx in candidates:
                gain = 0
                for t_idx in block_to_tsets[b_idx]:
                    if uncovered[t_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
                    
            # Add the best block
            current_blocks.append(best_block_idx)
            
            # Update uncovered status
            for t_idx in block_to_tsets[best_block_idx]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    num_uncovered -= 1
            
        # Local Search: Redundancy Removal
        # We have a list of block indices.
        # Check if any block is redundant.
        # cover_count for each t-set
        cover_count = [0] * num_t_sets
        for b_idx in current_blocks:
            for t_idx in block_to_tsets[b_idx]:
                cover_count[t_idx] += 1
                
        # Iteratively remove redundant blocks
        changed = True
        while changed:
            changed = False
            # Iterate backwards to remove from list easily
            for i in range(len(current_blocks) - 1, -1, -1):
                b_idx = current_blocks[i]
                # Check if all t-sets in this block are covered by others
                redundant = True
                for t_idx in block_to_tsets[b_idx]:
                    if cover_count[t_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block
                    for t_idx in block_to_tsets[b_idx]:
                        cover_count[t_idx] -= 1
                    current_blocks.pop(i)
                    changed = True
                    break # Restart scan from beginning after removal? Or continue?
                    # Breaking and restarting is safer for correctness of "changed" logic if we modify list.
                    # Actually, if we remove one, other blocks might become redundant.
                    # So restart the while loop.
        
        if len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = current_blocks[:]
            
    # Convert indices back to lists of points
    if best_blocks is None:
        # Fallback: return empty or a single block? 
        # Should not happen for valid parameters.
        return []
        
    return [list(all_blocks[b_idx]) for b_idx in best_blocks]
