import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # 2. Precompute all k-blocks and their covered t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # block_t_indices[i] is a list of indices of t-subsets covered by block i
    block_t_indices = []
    # t_sub_to_blocks[j] is a list of indices of blocks covering t-subset j
    t_sub_to_blocks = [[] for _ in range(num_t)]
    
    for i, block in enumerate(all_blocks):
        indices = []
        for sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[sub]
            indices.append(idx)
            t_sub_to_blocks[idx].append(i)
        block_t_indices.append(indices)

    # Helper: Fast Greedy Cover
    # Strategy: Pick a random uncovered t-subset, find the block covering it 
    # that covers the most other uncovered t-subsets.
    def fast_greedy():
        uncovered_mask = np.ones(num_t, dtype=bool)
        selected = []
        
        while uncovered_mask.any():
            if time.time() > deadline - 5.0:
                break
            
            # Pick a random uncovered t-subset
            uncovered_indices = np.where(uncovered_mask)[0]
            if len(uncovered_indices) == 0:
                break
            target_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
            
            # Get blocks covering this target
            candidate_blocks = t_sub_to_blocks[target_idx]
            
            best_gain = -1
            best_block = -1
            
            for b_idx in candidate_blocks:
                # Calculate gain: number of uncovered t-subsets covered by b_idx
                # We can use numpy sum on the mask intersected with block's t-subsets
                gain = np.sum(uncovered_mask[block_t_indices[b_idx]])
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block == -1:
                break # Should not happen
                
            selected.append(best_block)
            uncovered_mask[block_t_indices[best_block]] = False
            
        return selected

    # Local Search: Minimize number of blocks
    def local_search(selected):
        if not selected:
            return selected
            
        # Initialize counts
        counts = np.zeros(num_t, dtype=np.int32)
        for b in selected:
            counts[block_t_indices[b]] += 1
        
        # Remove redundant blocks
        changed = True
        while changed:
            changed = False
            # Shuffle to avoid bias
            random.shuffle(selected)
            for i in range(len(selected) - 1, -1, -1):
                if time.time() > deadline - 5.0:
                    return selected
                b = selected[i]
                # Check if any t-subset in b has count 1
                # If min count > 1, it's redundant
                if np.min(counts[block_t_indices[b]]) > 1:
                    # Remove b
                    counts[block_t_indices[b]] -= 1
                    selected.pop(i)
                    changed = True
        
        return selected

    # ILS Main Loop
    best_solution = None
    best_len = float('inf')
    
    # Initial Greedy
    current_selected = fast_greedy()
    current_selected = local_search(current_selected)
    
    if not current_selected:
        # Fallback: if greedy failed, just return a trivial cover or empty?
        # It shouldn't fail. But if it does, we need a valid output.
        # A valid cover is required.
        # If greedy fails, it means we didn't cover everything.
        # Let's try a simple full scan greedy as backup if fast greedy fails.
        pass

    best_solution = current_selected[:]
    best_len = len(current_selected)
    
    while time.time() < deadline - 2.0:
        # Kick: Remove a few random blocks
        if not current_selected:
            break
            
        kick_size = 2 + random.randint(0, 2) # 2-4 blocks
        # FIX: Clamp kick_size to valid range
        kick_size = min(kick_size, len(current_selected))
        
        if kick_size == 0:
            continue
            
        removed_blocks = random.sample(current_selected, kick_size)
        
        # Apply kick
        counts = np.zeros(num_t, dtype=np.int32)
        for b in current_selected:
            counts[block_t_indices[b]] += 1
            
        for b in removed_blocks:
            counts[block_t_indices[b]] -= 1
        
        # Now counts might have 0s (uncovered)
        uncovered_mask = (counts == 0)
        
        # We add blocks to current_selected (excluding removed ones? No, we can re-add them or others)
        # Let's just build a new list.
        new_selected = [b for b in current_selected if b not in removed_blocks]
        
        # Recover: Greedily add blocks to cover all 0s
        # Use the fast greedy logic on the remaining uncovered parts
        # But fast_greedy starts from scratch. We need a "repair" greedy.
        
        while uncovered_mask.any():
            if time.time() > deadline - 5.0:
                break
            uncovered_indices = np.where(uncovered_mask)[0]
            if len(uncovered_indices) == 0:
                break
            target_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
            
            candidate_blocks = t_sub_to_blocks[target_idx]
            
            best_gain = -1
            best_block = -1
            
            for b_idx in candidate_blocks:
                gain = np.sum(uncovered_mask[block_t_indices[b_idx]])
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block == -1:
                break # Stuck
                
            new_selected.append(best_block)
            idxs = block_t_indices[best_block]
            counts[idxs] += 1
            uncovered_mask[idxs] = False
            
        # Local Search on new_solution
        new_selected = local_search(new_selected)
        
        if len(new_selected) < best_len:
            best_len = len(new_selected)
            best_solution = new_selected[:]
            current_selected = new_selected[:]
        else:
            # With some probability, accept worse or equal to escape local minima?
            # Or just restart from best?
            current_selected = best_solution[:]

    if not best_solution:
        # Emergency fallback: return an empty list or a minimal valid set?
        # The task requires a valid cover. If we can't find one, we fail.
        # But fast_greedy should work.
        return []

    return [list(all_blocks[i]) for i in best_solution]
