import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s buffer
    
    # Precomputation
    # All t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    num_t_subs = len(all_t_subs)
    
    # All k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    # For each k-subset, store the indices of the t-subsets it contains
    k_sub_t_indices = []
    for k_sub in all_k_subs:
        t_indices = [t_sub_index[sub] for sub in itertools.combinations(k_sub, t)]
        k_sub_t_indices.append(t_indices)
        
    num_k_subs = len(all_k_subs)
    
    # Greedy Initialization
    cover_count = [0] * num_t_subs
    blocks = []
    
    # Helper to update coverage
    def add_block(idx):
        for t_idx in k_sub_t_indices[idx]:
            cover_count[t_idx] += 1
        blocks.append(idx)
        
    def remove_block(idx):
        blocks.remove(idx)
        for t_idx in k_sub_t_indices[idx]:
            cover_count[t_idx] -= 1
            
    # Greedy: pick block covering most uncovered
    # To speed up, we can just iterate and pick max gain
    # Gain of block i = number of t-sets in i with cover_count == 0
    while True:
        if time.time() > deadline:
            break
        best_gain = -1
        best_idx = -1
        # We can optimize this, but for v=19, num_k_subs=50k, it's fast enough per step
        # Number of steps in greedy is likely < 200
        for i in range(num_k_subs):
            gain = 0
            for t_idx in k_sub_t_indices[i]:
                if cover_count[t_idx] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = i
            if best_gain == len(k_sub_t_indices[i]): # Can't do better
                break
                
        if best_gain == 0:
            break # All covered
            
        add_block(best_idx)
        
    # Check if we are valid
    def is_valid():
        for c in cover_count:
            if c == 0:
                return False
        return True
        
    best_blocks = blocks[:] if is_valid() else []
    best_len = len(best_blocks)
    
    # Simulated Annealing
    if not is_valid():
        # Repair greedy if it failed due to time? Unlikely.
        pass
        
    current_blocks = blocks[:]
    current_cover = cover_count[:]
    
    # Energy function
    def get_energy(blocks, cover):
        uncovered = sum(1 for c in cover if c == 0)
        # We want to minimize blocks, but must be valid.
        # If invalid, heavy penalty.
        return len(blocks) + 1000 * uncovered
        
    current_energy = get_energy(current_blocks, current_cover)
    
    T = 1.0
    cooling = 0.999
    min_T = 0.001
    
    # Precompute random access for k-subsets
    # We need to pick a random k-subset not in current blocks
    # To make this efficient, we can just pick a random index and check if it's in the set
    # Using a set for current blocks for O(1) lookup
    current_block_set = set(current_blocks)
    
    def get_uncovered_count(cover):
        return sum(1 for c in cover if c == 0)
        
    current_uncovered = get_uncovered_count(current_cover)
    
    while time.time() < deadline:
        # Pick a move
        move_type = random.random()
        
        if move_type < 0.4: # Remove
            if not current_blocks:
                continue
            idx_to_remove = random.choice(current_blocks)
            # Apply
            for t_idx in k_sub_t_indices[idx_to_remove]:
                current_cover[t_idx] -= 1
            current_blocks.remove(idx_to_remove)
            current_block_set.remove(idx_to_remove)
            
            new_uncovered = get_uncovered_count(current_cover)
            new_energy = get_energy(current_blocks, current_cover)
            
            delta = new_energy - current_energy
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                current_energy = new_energy
                current_uncovered = new_uncovered
                if current_uncovered == 0 and len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks[:]
            else:
                # Revert
                current_blocks.append(idx_to_remove)
                current_block_set.add(idx_to_remove)
                for t_idx in k_sub_t_indices[idx_to_remove]:
                    current_cover[t_idx] += 1
                
        elif move_type < 0.8: # Add
            # Pick random block not in set
            # Simple retry loop
            while True:
                cand_idx = random.randint(0, num_k_subs - 1)
                if cand_idx not in current_block_set:
                    break
                    
            # Apply
            for t_idx in k_sub_t_indices[cand_idx]:
                current_cover[t_idx] += 1
            current_blocks.append(cand_idx)
            current_block_set.add(cand_idx)
            
            new_uncovered = get_uncovered_count(current_cover)
            new_energy = get_energy(current_blocks, current_cover)
            
            delta = new_energy - current_energy
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                current_energy = new_energy
                current_uncovered = new_uncovered
                if current_uncovered == 0 and len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks[:]
            else:
                # Revert
                current_blocks.remove(cand_idx)
                current_block_set.remove(cand_idx)
                for t_idx in k_sub_t_indices[cand_idx]:
                    current_cover[t_idx] -= 1
        else: # Swap
            if not current_blocks:
                continue
            idx_to_swap = random.choice(current_blocks)
            while True:
                cand_idx = random.randint(0, num_k_subs - 1)
                if cand_idx not in current_block_set:
                    break
                    
            # Apply Swap
            # Remove old
            for t_idx in k_sub_t_indices[idx_to_swap]:
                current_cover[t_idx] -= 1
            # Add new
            for t_idx in k_sub_t_indices[cand_idx]:
                current_cover[t_idx] += 1
            
            current_blocks.remove(idx_to_swap)
            current_block_set.remove(idx_to_swap)
            current_blocks.append(cand_idx)
            current_block_set.add(cand_idx)
            
            new_uncovered = get_uncovered_count(current_cover)
            new_energy = get_energy(current_blocks, current_cover)
            
            delta = new_energy - current_energy
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                current_energy = new_energy
                current_uncovered = new_uncovered
                if current_uncovered == 0 and len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks[:]
            else:
                # Revert
                # Remove new
                for t_idx in k_sub_t_indices[cand_idx]:
                    current_cover[t_idx] -= 1
                # Add old
                for t_idx in k_sub_t_indices[idx_to_swap]:
                    current_cover[t_idx] += 1
                
                current_blocks.remove(cand_idx)
                current_block_set.remove(cand_idx)
                current_blocks.append(idx_to_swap)
                current_block_set.add(idx_to_swap)
                
        T *= cooling
        if T < min_T:
            T = min_T
            
    # Final check: if best_blocks is empty, return current if valid, else greedy
    if not best_blocks:
        if is_valid():
            best_blocks = current_blocks[:]
        else:
            # Fallback to greedy if SA didn't find valid (should not happen)
            # Re-run greedy quickly? Or just return current if it's close?
            # The task requires valid.
            # Let's ensure we return a valid cover.
            # If SA lost validity, we can greedily repair best_blocks
            # But best_blocks was only updated if valid.
            # If best_blocks is empty, it means we never had a valid cover after init?
            # Init was valid. So best_blocks should not be empty.
            pass
            
    # Convert indices back to lists
    result = [list(all_k_subs[i]) for i in best_blocks]
    
    # Sort blocks and points for consistency (not strictly required but good)
    for b in result:
        b.sort()
    result.sort()
        
    return result
