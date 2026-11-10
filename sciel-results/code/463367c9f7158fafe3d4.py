import itertools
import time
import random
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Precompute t-subsets
    all_t = list(itertools.combinations(range(v), t))
    t_index = {ts: i for i, ts in enumerate(all_t)}
    n_t = len(all_t)
    full_mask = (1 << n_t) - 1

    # Precompute k-subsets and their masks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    block_masks = []
    # Also store list of t-subset indices for each block for fast cov_count updates
    block_t_indices = []
    for block in all_blocks:
        mask = 0
        indices = []
        for ts in itertools.combinations(block, t):
            idx = t_index[ts]
            mask |= (1 << idx)
            indices.append(idx)
        block_masks.append(mask)
        block_t_indices.append(indices)

    def greedy_cover(rng, deadline):
        """
        Returns list of block indices forming a valid cover.
        """
        uncovered = full_mask
        blocks = []
        
        while uncovered:
            if time.time() > deadline:
                return None
            
            max_gain = 0
            best_block = -1
            # To allow randomization, we can pick randomly among max gainers
            # But scanning 346k blocks twice is slow.
            # Let's just scan once, track max, and collect candidates if tie?
            # Or just pick first max? 
            # For diversity in restarts, let's add noise or random tie break.
            
            candidates = []
            for b_idx in range(n_blocks):
                g = (block_masks[b_idx] & uncovered).bit_count()
                if g > max_gain:
                    max_gain = g
                    candidates = [b_idx]
                elif g == max_gain and g > 0:
                    candidates.append(b_idx)
            
            if max_gain == 0:
                return None # Should not happen
            
            best_block = rng.choice(candidates)
            blocks.append(best_block)
            uncovered &= ~block_masks[best_block]
            
        return blocks

    def remove_redundant(blocks, deadline):
        """
        Removes blocks that do not uniquely cover any t-subset.
        Returns modified list of block indices.
        """
        if not blocks:
            return blocks
        
        # Compute coverage counts
        cov_count = [0] * n_t
        for b_idx in blocks:
            for idx in block_t_indices[b_idx]:
                cov_count[idx] += 1
        
        changed = True
        while changed:
            changed = False
            # Iterate in reverse to handle popping
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline:
                    return blocks
                
                b_idx = blocks[i]
                redundant = True
                for idx in block_t_indices[b_idx]:
                    if cov_count[idx] == 1:
                        redundant = False
                        break
                
                if redundant:
                    blocks.pop(i)
                    for idx in block_t_indices[b_idx]:
                        cov_count[idx] -= 1
                    changed = True
        
        return blocks

    def simulate_annealing(initial_blocks, deadline):
        """
        Optimizes the block set using Simulated Annealing.
        Cost = len(blocks) + LAMBDA * (n_t - covered_count)
        We track cov_count to calculate covered_count quickly?
        Actually, we track uncovered_count.
        Cost = len(blocks) + LAMBDA * uncovered_count
        """
        LAMBDA = 100 # Penalty for uncovered triples
        
        # State
        current_blocks = set(initial_blocks)
        # cov_count[i] is number of blocks in current_blocks covering triple i
        cov_count = [0] * n_t
        for b_idx in current_blocks:
            for idx in block_t_indices[b_idx]:
                cov_count[idx] += 1
        
        # Calculate initial uncovered count
        uncovered_count = 0
        for c in cov_count:
            if c == 0:
                uncovered_count += 1
        
        current_cost = len(current_blocks) + LAMBDA * uncovered_count
        best_blocks = list(current_blocks)
        best_cost = current_cost
        
        # SA Parameters
        T = 50.0
        cooling = 0.995
        min_T = 0.1
        
        # Precompute list of all block indices for random selection
        all_indices = list(range(n_blocks))
        
        while T > min_T and time.time() < deadline - 5.0:
            # Choose move
            # If invalid (uncovered_count > 0), prioritize Add (Repair)
            # If valid, prioritize Remove (Optimize) or Random Add (Perturb)
            
            if uncovered_count > 0:
                # Repair: Find best block to add
                # We want to minimize uncovered_count, so maximize gain on uncovered.
                # But scanning all 346k is slow.
                # Let's pick a random block? No, that's slow to converge.
                # Let's pick a block that covers a random uncovered triple?
                # Or just do a greedy add step?
                
                # Efficient Repair: Pick a random uncovered triple, pick a random block covering it.
                # This is O(1) roughly.
                # Find an uncovered triple
                # We don't have a list of uncovered triples, only count.
                # Let's just pick a random block from all_blocks?
                # No, that's inefficient.
                
                # Let's stick to Greedy Add for repair, but limit candidates?
                # Actually, if we are in SA, we can accept worse moves.
                # But if we are invalid, we MUST get valid eventually.
                # Let's use a "Greedy Add" move here.
                
                # To speed up, we can sample 100 random blocks and pick the best one.
                sample_size = 100
                best_gain = -1
                best_b = -1
                for _ in range(sample_size):
                    b_idx = random.choice(all_indices)
                    if b_idx in current_blocks:
                        continue
                    # Calculate gain: how many uncovered triples does b_idx cover?
                    # gain = sum(1 for idx in block_t_indices[b_idx] if cov_count[idx] == 0)
                    gain = 0
                    for idx in block_t_indices[b_idx]:
                        if cov_count[idx] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_b = b_idx
                
                if best_b != -1:
                    # Apply move: Add best_b
                    old_uncovered = uncovered_count
                    for idx in block_t_indices[best_b]:
                        if cov_count[idx] == 0:
                            uncovered_count -= 1
                        cov_count[idx] += 1
                    
                    new_cost = len(current_blocks) + 1 + LAMBDA * uncovered_count
                    delta_cost = new_cost - current_cost
                    
                    # Always accept repair moves? Or SA accept?
                    # Since we are invalid, we want to fix it.
                    # delta_cost should be negative (uncovered drops by best_gain, cost drops by LAMBDA*best_gain - 1)
                    # If LAMBDA is large, this is always an improvement.
                    current_blocks.add(best_b)
                    current_cost = new_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_blocks = list(current_blocks)
                
            else:
                # Valid state. Try to improve (Remove) or Perturb (Add/Remove)
                # 80% chance Remove, 20% chance Add (Perturb)
                
                if len(current_blocks) > 0 and random.random() < 0.8:
                    # Remove Move
                    # Pick random block
                    b_idx = random.choice(list(current_blocks))
                    
                    # Check if removal causes uncovered
                    # We need to know if any triple in b_idx has cov_count == 1
                    delta_uncovered = 0
                    for idx in block_t_indices[b_idx]:
                        if cov_count[idx] == 1:
                            delta_uncovered += 1
                    
                    # Apply tentatively
                    for idx in block_t_indices[b_idx]:
                        cov_count[idx] -= 1
                        if cov_count[idx] == 0:
                            uncovered_count += 1
                    
                    new_cost = len(current_blocks) - 1 + LAMBDA * uncovered_count
                    delta_cost = new_cost - current_cost
                    
                    # SA Acceptance
                    if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                        current_blocks.remove(b_idx)
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_blocks = list(current_blocks)
                    else:
                        # Reject: Undo
                        for idx in block_t_indices[b_idx]:
                            if cov_count[idx] == 0:
                                uncovered_count -= 1
                            cov_count[idx] += 1
                            # Note: uncovered_count was incremented above if it hit 0.
                            # So we decrement if it is now 0.
                else:
                    # Add Move (Perturb)
                    # Pick random block not in current_blocks
                    b_idx = random.choice(all_indices)
                    if b_idx in current_blocks:
                        continue
                    
                    # Calculate gain (reduction in uncovered, which is 0)
                    # So delta_uncovered = 0
                    # delta_cost = 1 + LAMBDA * 0 = 1
                    
                    # Apply tentatively
                    for idx in block_t_indices[b_idx]:
                        cov_count[idx] += 1
                        # uncovered_count unchanged (was 0, stays 0)
                    
                    new_cost = len(current_blocks) + 1 + LAMBDA * uncovered_count
                    delta_cost = new_cost - current_cost # +1
                    
                    # SA Acceptance
                    if random.random() < math.exp(-delta_cost / T):
                        current_blocks.add(b_idx)
                        current_cost = new_cost
                        # Cost increased, so best_cost unchanged
                    else:
                        # Reject: Undo
                        for idx in block_t_indices[b_idx]:
                            cov_count[idx] -= 1
            
            T *= cooling
        
        return best_blocks

    # Main Loop
    best_blocks = None
    best_len = float('inf')
    
    restart_count = 0
    while time.time() < deadline - 10.0:
        restart_count += 1
        rng = random.Random(42 + restart_count)
        
        # 1. Greedy Init
        blocks = greedy_cover(rng, deadline)
        if blocks is None:
            continue
        
        # 2. Remove Redundant
        blocks = remove_redundant(blocks, deadline)
        
        current_len = len(blocks)
        if current_len < best_len:
            best_len = current_len
            best_blocks = [all_blocks[b_idx] for b_idx in blocks]
        
        # 3. Simulated Annealing
        if time.time() < deadline - 20.0:
            sa_blocks = simulate_annealing(blocks, deadline)
            if sa_blocks is not None:
                sa_len = len(sa_blocks)
                if sa_len < best_len:
                    best_len = sa_len
                    best_blocks = [all_blocks[b_idx] for b_idx in sa_blocks]

    if best_blocks is None:
        # Fallback
        blocks = greedy_cover(random.Random(0), deadline)
        if blocks:
            blocks = remove_redundant(blocks, deadline)
            best_blocks = [all_blocks[b_idx] for b_idx in blocks]
        else:
            return []
            
    return best_blocks
