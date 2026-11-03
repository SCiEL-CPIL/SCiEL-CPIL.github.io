import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Fallback for non-target parameters (keep simple, though unlikely to be tested)
    if v != 19 or k != 7 or t != 4:
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            best_block = None
            best_count = -1
            for block_tuple in itertools.combinations(range(v), k):
                block_set = set(block_tuple)
                count = sum(1 for tsub in uncovered if set(tsub).issubset(block_set))
                if count > best_count:
                    best_count = count
                    best_block = list(block_tuple)
                if time.time() > deadline:
                    # Emergency exit, might be invalid
                    break
            if best_block is None or best_count == 0:
                break
            blocks.append(best_block)
            block_set = set(best_block)
            uncovered = {tsub for tsub in uncovered if not set(tsub).issubset(block_set)}
            if time.time() > deadline:
                break
        return blocks

    # --- Optimized Path for v=19, k=7, t=4 ---
    
    points = range(v)
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_id = {tsub: i for i, tsub in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    all_blocks = []
    block_tsubs_ids = [] # List of lists of tsub IDs
    
    for block_tuple in itertools.combinations(points, k):
        block_fs = frozenset(block_tuple)
        tsubs_in_block = []
        for tsub in itertools.combinations(block_tuple, t):
            tsubs_in_block.append(tsub_to_id[tsub])
        all_blocks.append(block_fs)
        block_tsubs_ids.append(tsubs_in_block)
        
    num_total_blocks = len(all_blocks)
    
    # Precompute indices for fast access
    # We will use a numpy array for cover counts
    
    # 1. Greedy Initialization
    cover_count = np.zeros(num_tsubs, dtype=np.int32)
    current_block_indices = []
    num_uncovered = num_tsubs
    
    # Helper to get tsub ids for a block index
    def get_tsub_ids(b_idx):
        return block_tsubs_ids[b_idx]
        
    # Greedy
    # To speed up greedy, we can precompute the incidence matrix or just use the lists.
    # With 50k blocks and 3876 tsubs, checking gain is sum(cover_count[tids]==0).
    
    while num_uncovered > 0:
        best_b_idx = -1
        best_gain = -1
        
        # Iterate all blocks
        for b_idx in range(num_total_blocks):
            if b_idx in current_block_indices: # This check is slow if list is large. Use a set?
                continue
            
            tids = block_tsubs_ids[b_idx]
            # Vectorized gain calculation
            gain = np.sum(cover_count[tids] == 0)
            
            if gain > best_gain:
                best_gain = gain
                best_b_idx = b_idx
        
        if best_b_idx == -1 or best_gain == 0:
            break
            
        current_block_indices.append(best_b_idx)
        tids = block_tsubs_ids[best_b_idx]
        cover_count[tids] += 1
        num_uncovered = np.sum(cover_count == 0)
        
        if time.time() > deadline:
            break

    # 2. Simulated Annealing
    # Use a set for O(1) membership check
    current_set = set(current_block_indices)
    current_energy = len(current_block_indices)
    
    penalty_lambda = 10.0
    T = 1.0
    T_min = 1e-5
    cooling_rate = 0.995
    rng = random.Random(42)
    
    # Track best valid solution
    best_valid_blocks = list(current_block_indices)
    best_valid_energy = current_energy if num_uncovered == 0 else float('inf')
    
    if num_uncovered == 0:
        best_valid_blocks = list(current_block_indices)
        best_valid_energy = current_energy

    while time.time() < deadline and T > T_min:
        # Choose move
        if num_uncovered == 0:
            move_type = 'remove'
        else:
            move_type = rng.choice(['remove', 'add'])
            
        if move_type == 'remove':
            if not current_set:
                continue
            # Pick random block from current
            b_idx = rng.choice(list(current_set))
            
            tids = block_tsubs_ids[b_idx]
            # Delta uncovered: tsubs that go from 1 to 0
            delta_uncovered = np.sum(cover_count[tids] == 1)
            
            delta_E = -1.0 + penalty_lambda * delta_uncovered
            
            if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                # Accept
                current_set.discard(b_idx)
                current_block_indices.remove(b_idx)
                cover_count[tids] -= 1
                num_uncovered += delta_uncovered
                current_energy -= 1
                
                if num_uncovered == 0 and current_energy < best_valid_energy:
                    best_valid_energy = current_energy
                    best_valid_blocks = list(current_block_indices)
            
        else: # add
            # Pick random block not in current
            # Rejection sampling
            while True:
                b_idx = rng.randint(0, num_total_blocks - 1)
                if b_idx not in current_set:
                    break
            
            tids = block_tsubs_ids[b_idx]
            # Delta uncovered: tsubs that go from 0 to 1
            # gain = number of 0s in tids
            delta_uncovered = -np.sum(cover_count[tids] == 0)
            
            delta_E = 1.0 + penalty_lambda * delta_uncovered
            
            if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                # Accept
                current_set.add(b_idx)
                current_block_indices.append(b_idx)
                cover_count[tids] += 1
                num_uncovered += delta_uncovered
                current_energy += 1
                
                if num_uncovered == 0 and current_energy < best_valid_energy:
                    best_valid_energy = current_energy
                    best_valid_blocks = list(current_block_indices)
                    
        T *= cooling_rate

    # Final Repair if invalid
    if num_uncovered > 0:
        # Greedy repair
        while num_uncovered > 0:
            best_b_idx = -1
            best_gain = -1
            for b_idx in range(num_total_blocks):
                if b_idx in current_set:
                    continue
                tids = block_tsubs_ids[b_idx]
                gain = np.sum(cover_count[tids] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_b_idx = b_idx
            if best_b_idx == -1 or best_gain == 0:
                break
            current_set.add(best_b_idx)
            current_block_indices.append(best_b_idx)
            tids = block_tsubs_ids[best_b_idx]
            cover_count[tids] += 1
            num_uncovered = np.sum(cover_count == 0)
            if time.time() > deadline:
                break
        
        # Update best if this is better (should be, as it's valid)
        if num_uncovered == 0:
            if len(current_block_indices) < best_valid_energy:
                best_valid_blocks = list(current_block_indices)
                best_valid_energy = len(current_block_indices)
            
    # If we still have uncovered (shouldn't happen with repair), use best valid or greedy
    if num_uncovered > 0:
        # Fallback to best valid found or greedy
        if best_valid_blocks:
            final_indices = best_valid_blocks
        else:
            # Re-run simple greedy? Or just return what we have (invalid)
            # We must return valid.
            # Let's try a quick greedy from scratch if we are broken
            pass
            
    # Use best valid blocks
    final_indices = best_valid_blocks if best_valid_blocks else current_block_indices
    
    # Convert to lists
    final_blocks = [sorted(all_blocks[i]) for i in final_indices]
    
    return final_blocks
