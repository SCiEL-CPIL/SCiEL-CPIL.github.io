import itertools
import random
import time

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using greedy construction with local search.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precomputation for bitmask efficiency
    # Only efficient for v < 60
    if v > 30:
        # Fallback to slower generic greedy if v is large (unlikely for this task)
        return _generic_greedy(v, k, t)

    all_t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_t_subsets)
    
    # Map mask to index
    mask_to_idx = {}
    for i, subset in enumerate(all_t_subsets):
        mask = 0
        for p in subset:
            mask |= (1 << p)
        mask_to_idx[mask] = i
        
    # Precompute masks for all t-subsets
    subset_masks = [0] * n_subsets
    for i, subset in enumerate(all_t_subsets):
        mask = 0
        for p in subset:
            mask |= (1 << p)
        subset_masks[i] = mask

    # Helper to get t-subsets of a block (mask)
    def get_t_subsets_of_block(block_mask):
        # Extract points
        points = [p for p in range(v) if (block_mask >> p) & 1]
        return list(itertools.combinations(points, t))

    # Greedy cover for a specific set of uncovered subset indices
    def greedy_cover(uncovered_indices):
        if not uncovered_indices:
            return []
        
        local_uncovered = set(uncovered_indices)
        local_blocks = []
        
        while local_uncovered:
            # Pick a random uncovered subset
            idx = random.choice(list(local_uncovered))
            S_mask = subset_masks[idx]
            
            # Start block with points from S
            # We need points, not just mask, for combination generation
            S_points = [p for p in range(v) if (S_mask >> p) & 1]
            block_points = list(S_points)
            
            while len(block_points) < k:
                best_x = -1
                best_gain = -1
                
                # Candidates: all points not in block
                block_set = set(block_points)
                candidates = [x for x in range(v) if x not in block_set]
                
                for x in candidates:
                    gain = 0
                    # Check all (t-1) subsets of current block
                    # If adding x completes an uncovered t-subset, gain++
                    # A t-subset U is covered by B U {x} iff U \ {x} is a subset of B
                    # So we iterate (t-1) subsets Q of B, and check if Q U {x} is in local_uncovered
                    for Q in itertools.combinations(block_points, t - 1):
                        Q_mask = 0
                        for p in Q:
                            Q_mask |= (1 << p)
                        U_mask = Q_mask | (1 << x)
                        # Check if U_mask is in local_uncovered
                        # We need to map U_mask to index to check local_uncovered
                        # local_uncovered contains indices.
                        # We can check if mask_to_idx[U_mask] is in local_uncovered
                        # But U_mask might not be a valid t-subset mask? 
                        # Yes it is, because Q is size t-1 and x is new.
                        u_idx = mask_to_idx.get(U_mask)
                        if u_idx is not None and u_idx in local_uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x != -1:
                    block_points.append(best_x)
                else:
                    # Should not happen if there are uncovered subsets
                    break
            
            block_mask = 0
            for p in block_points:
                block_mask |= (1 << p)
            
            local_blocks.append(block_mask)
            
            # Remove covered subsets from local_uncovered
            to_remove = []
            for u_idx in local_uncovered:
                u_mask = subset_masks[u_idx]
                if (u_mask & block_mask) == u_mask:
                    to_remove.append(u_idx)
            
            for u_idx in to_remove:
                local_uncovered.remove(u_idx)
                
        return local_blocks

    # Main Logic
    # 1. Generate initial covering
    best_blocks = None
    best_len = float('inf')
    
    # Run greedy multiple times to find a good initial solution
    num_restarts = 50
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
            
        # All subsets are uncovered
        all_indices = set(range(n_subsets))
        blocks = greedy_cover(all_indices)
        
        # Reduction pass
        # Compute counts
        counts = [0] * n_subsets
        for b_mask in blocks:
            for subset in get_t_subsets_of_block(b_mask):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                idx = mask_to_idx[mask]
                counts[idx] += 1
        
        # Try to remove blocks
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                b_mask = blocks[i]
                # Decrement counts
                for subset in get_t_subsets_of_block(b_mask):
                    mask = 0
                    for p in subset:
                        mask |= (1 << p)
                    idx = mask_to_idx[mask]
                    counts[idx] -= 1
                    if counts[idx] < 0:
                        # Error, shouldn't happen
                        pass
                
                # Check if any count is 0
                is_critical = False
                for subset in get_t_subsets_of_block(b_mask):
                    mask = 0
                    for p in subset:
                        mask |= (1 << p)
                    idx = mask_to_idx[mask]
                    if counts[idx] == 0:
                        is_critical = True
                        break
                
                if not is_critical:
                    # Remove block
                    blocks.pop(i)
                    changed = True
                else:
                    # Restore counts
                    for subset in get_t_subsets_of_block(b_mask):
                        mask = 0
                        for p in subset:
                            mask |= (1 << p)
                        idx = mask_to_idx[mask]
                        counts[idx] += 1
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]
            
    # 2. Local Search (Simulated Annealing / Hill Climbing)
    if best_blocks is not None and time.time() < deadline:
        current_blocks = best_blocks[:]
        current_len = len(current_blocks)
        
        # Compute counts for current blocks
        counts = [0] * n_subsets
        for b_mask in current_blocks:
            for subset in get_t_subsets_of_block(b_mask):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                idx = mask_to_idx[mask]
                counts[idx] += 1
        
        temperature = 1.0
        cooling = 0.995
        
        while time.time() < deadline and temperature > 0.01:
            if not current_blocks:
                break
                
            # Pick a random block
            i = random.randint(0, len(current_blocks) - 1)
            b_mask = current_blocks[i]
            
            # Decrement counts for b_mask
            affected_indices = []
            for subset in get_t_subsets_of_block(b_mask):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                idx = mask_to_idx[mask]
                counts[idx] -= 1
                affected_indices.append(idx)
                
            # Find uncovered indices (count became 0)
            uncovered = [idx for idx in affected_indices if counts[idx] == 0]
            
            if not uncovered:
                # Block was redundant
                current_blocks.pop(i)
                current_len -= 1
                # Temperature can stay or cool slightly
                temperature *= cooling
            else:
                # Need to cover 'uncovered'
                new_blocks = greedy_cover(uncovered)
                
                # Filter out blocks already in current_blocks
                existing_masks = set(current_blocks)
                new_blocks_filtered = [b for b in new_blocks if b not in existing_masks]
                
                # Tentatively add new_blocks
                for nb in new_blocks_filtered:
                    for subset in get_t_subsets_of_block(nb):
                        mask = 0
                        for p in subset:
                            mask |= (1 << p)
                        idx = mask_to_idx[mask]
                        counts[idx] += 1
                        
                new_len = len(current_blocks) - 1 + len(new_blocks_filtered)
                
                if new_len < current_len:
                    # Accept
                    current_blocks.pop(i)
                    current_blocks.extend(new_blocks_filtered)
                    current_len = new_len
                    temperature *= cooling
                elif new_len == current_len and random.random() < temperature:
                    # Accept (plateau move)
                    current_blocks.pop(i)
                    current_blocks.extend(new_blocks_filtered)
                    # current_len unchanged
                    temperature *= cooling
                else:
                    # Revert
                    for nb in new_blocks_filtered:
                        for subset in get_t_subsets_of_block(nb):
                            mask = 0
                            for p in subset:
                                mask |= (1 << p)
                            idx = mask_to_idx[mask]
                            counts[idx] -= 1
                            
                    # Restore b_mask counts
                    for idx in affected_indices:
                        counts[idx] += 1
                        
                    temperature *= cooling

    # Convert masks back to lists of points
    final_blocks = []
    for b_mask in best_blocks:
        points = [p for p in range(v) if (b_mask >> p) & 1]
        final_blocks.append(sorted(points))
        
    return final_blocks

def _generic_greedy(v, k, t):
    """Fallback for larger v"""
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks
