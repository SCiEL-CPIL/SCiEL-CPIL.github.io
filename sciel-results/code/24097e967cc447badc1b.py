import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    # Precompute t-subsets
    all_t = list(itertools.combinations(range(v), t))
    t_index = {ts: i for i, ts in enumerate(all_t)}
    n_t = len(all_t)
    full_mask = (1 << n_t) - 1

    # Precompute k-subsets and their masks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    block_masks = []
    for block in all_blocks:
        mask = 0
        for ts in itertools.combinations(block, t):
            mask |= (1 << t_index[ts])
        block_masks.append(mask)

    def greedy_cover(rng, deadline):
        uncovered = full_mask
        blocks = []
        
        while uncovered:
            if time.time() > deadline:
                return None
            
            best_gain = 0
            best_block = -1
            # Collect top candidates to allow randomization
            # For speed, just find max and break ties randomly?
            # Or find top 3?
            # Let's stick to max, but randomize the order of checking?
            # No, shuffling 346k is slow.
            # We can just pick the first max found, but that's deterministic.
            # To randomize, we can add a small random noise to the gain?
            # Or just pick randomly among all blocks with max gain.
            
            # Optimization: If we just scan linearly, we get the first max.
            # To get a random max, we can count how many have max gain?
            # Too slow.
            
            # Alternative: Randomly permute the block indices ONCE at the start?
            # No, the "best" block changes as uncovered changes.
            
            # Let's just scan all. It's fast enough.
            # To introduce randomness, we can randomly skip some blocks? No.
            # We can randomly select a subset of blocks to consider?
            # No, we might miss the best one.
            
            # Simple randomization: 
            # Pick a random start index and wrap around? No.
            # Just scan. The determinism is handled by multiple restarts with different seeds?
            # But the greedy is deterministic given the order.
            # So we MUST randomize the choice among ties.
            
            # Strategy: 
            # 1. Find max_gain.
            # 2. Find all blocks with gain == max_gain.
            # 3. Pick one randomly.
            # This requires two passes or storing candidates.
            # Pass 1: Find max_gain.
            # Pass 2: Pick random from those with max_gain.
            # This doubles the time per step. 2.4s -> 4.8s. Still ok.
            
            max_gain = 0
            for b_idx in range(n_blocks):
                g = (block_masks[b_idx] & uncovered).bit_count()
                if g > max_gain:
                    max_gain = g
            
            if max_gain == 0:
                return None # Should not happen if uncovered is not 0 and blocks exist
            
            # Find candidates with max_gain
            candidates = []
            for b_idx in range(n_blocks):
                if (block_masks[b_idx] & uncovered).bit_count() == max_gain:
                    candidates.append(b_idx)
            
            best_block = rng.choice(candidates)
            blocks.append(best_block)
            uncovered &= ~block_masks[best_block]
            
        return blocks

    def remove_redundant(blocks, deadline):
        if not blocks:
            return blocks
        
        # Map block index to mask
        # blocks is a list of indices into all_blocks/block_masks
        
        # Compute coverage counts
        cov_count = [0] * n_t
        for b_idx in blocks:
            mask = block_masks[b_idx]
            m = mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                cov_count[idx] += 1
                m ^= lsb
        
        changed = True
        while changed:
            changed = False
            # Try to remove blocks
            # Iterate in reverse to handle popping
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline:
                    return blocks
                
                b_idx = blocks[i]
                mask = block_masks[b_idx]
                
                # Check if any t-subset is uniquely covered
                m = mask
                redundant = True
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    if cov_count[idx] == 1:
                        redundant = False
                        break
                    m ^= lsb
                
                if redundant:
                    blocks.pop(i)
                    m = mask
                    while m:
                        lsb = m & (-m)
                        idx = lsb.bit_length() - 1
                        cov_count[idx] -= 1
                        m ^= lsb
                    changed = True
        
        return blocks

    # Main Loop
    best_blocks = None
    best_len = float('inf')
    
    restart_count = 0
    while time.time() < deadline - 5.0:
        restart_count += 1
        rng = random.Random(42 + restart_count)
        
        blocks = greedy_cover(rng, deadline)
        if blocks is None:
            continue
            
        # Convert indices to actual blocks for return, but keep indices for internal processing?
        # The function returns list of blocks (lists of ints).
        # But for efficiency in local search, we work with indices.
        
        blocks = remove_redundant(blocks, deadline)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [all_blocks[b_idx] for b_idx in blocks]
            
        # Optional: Local Search Step (Perturb and Repair)
        # If time allows, try to improve further
        if time.time() < deadline - 10.0:
             # Pick a random block to remove
             if blocks:
                 remove_idx = rng.choice(range(len(blocks)))
                 perturbed_blocks = blocks[:]
                 perturbed_blocks.pop(remove_idx)
                 
                 # Repair: Greedily add blocks to cover what was lost
                 # We need to know what is uncovered now.
                 # Re-calculate uncovered mask for perturbed_blocks
                 uncovered = full_mask
                 for b_idx in perturbed_blocks:
                     uncovered &= ~block_masks[b_idx]
                 
                 # Greedily fill
                 while uncovered:
                     if time.time() > deadline:
                         break
                     max_gain = 0
                     for b_idx in range(n_blocks):
                         g = (block_masks[b_idx] & uncovered).bit_count()
                         if g > max_gain:
                             max_gain = g
                     if max_gain == 0: break
                     candidates = [b_idx for b_idx in range(n_blocks) if (block_masks[b_idx] & uncovered).bit_count() == max_gain]
                     new_b = rng.choice(candidates)
                     perturbed_blocks.append(new_b)
                     uncovered &= ~block_masks[new_b]
                 
                 perturbed_blocks = remove_redundant(perturbed_blocks, deadline)
                 
                 if len(perturbed_blocks) < best_len:
                     best_len = len(perturbed_blocks)
                     best_blocks = [all_blocks[b_idx] for b_idx in perturbed_blocks]
                     # Update current best for next perturbation?
                     # Let's stick to the best found so far for the next restart base?
                     # Or just keep doing restarts.
                     
    if best_blocks is None:
        # Fallback: Deterministic greedy
        blocks = greedy_cover(random.Random(0), deadline)
        if blocks:
             blocks = remove_redundant(blocks, deadline)
             best_blocks = [all_blocks[b_idx] for b_idx in blocks]
        else:
             return []
             
    return best_blocks
