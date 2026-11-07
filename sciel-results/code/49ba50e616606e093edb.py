# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 12, 'm': 40}

def _count_comparable_pairs_masks(masks):
    """
    Count the number of unordered pairs (i, j) with i < j such that 
    masks[i] is a strict subset of masks[j] or masks[j] is a strict subset of masks[i].
    """
    m = len(masks)
    count = 0
    for i in range(m):
        mi = masks[i]
        for j in range(i + 1, m):
            mj = masks[j]
            if mi == mj:
                continue
            # Check if mi is subset of mj
            if (mi & mj) == mi:
                count += 1
            # Check if mj is subset of mi
            elif (mi & mj) == mj:
                count += 1
    return count

def _masks_to_sets(masks, n):
    sets = []
    for mask in masks:
        s = []
        for b in range(n):
            if mask & (1 << b):
                s.append(b)
        sets.append(s)
    return sets

def construct():
    n = P['n']
    m = P['m']
    
    # Deadline guard: 85s safety margin for 90s limit
    start_time = time.time()
    max_time = 85.0
    
    # Strategy:
    # 1. Generate multiple diverse initial solutions based on random permutations (chains).
    # 2. Use Simulated Annealing (SA) to optimize each one.
    # 3. Keep the best result.
    
    best_overall_score = -1
    best_overall_masks = []
    
    # Number of restarts
    # Each restart involves SA for a certain time.
    # Let's do 5-10 restarts.
    
    restarts = 8
    
    for r in range(restarts):
        # Check time
        if time.time() - start_time > max_time - 1.0:
            break
        
        # 1. Initialization: Chain-based
        # Pick a random permutation of 12 elements.
        perm = list(range(n))
        random.shuffle(perm)
        
        # Create a base chain: C_k = {perm[0], ..., perm[k-1]} for k=0..12
        # We have 13 sets in this chain.
        # We need 40 sets.
        # Let's fill the remaining 27 sets with random subsets that are likely comparable to the chain.
        # Or just random subsets, and let SA fix them.
        
        init_masks = []
        # Add the chain sets
        chain_masks = [0] * (n + 1)
        for k in range(n + 1):
            mask = 0
            for i in range(k):
                mask |= (1 << perm[i])
            chain_masks[k] = mask
        
        # Add 7 of these chain sets (random selection to break symmetry slightly, or just all)
        # Actually, adding all 13 gives a strong backbone.
        init_masks.extend(chain_masks)
        
        # Need 40 - 13 = 27 more.
        # Generate random subsets.
        # To encourage comparability, maybe sample from sizes that are close to the chain sizes?
        # Or just uniform random.
        
        while len(init_masks) < m:
            size = random.randint(0, n)
            mask = 0
            # Pick 'size' random elements
            elems = random.sample(range(n), size)
            for e in elems:
                mask |= (1 << e)
            
            # Ensure distinctness
            if mask in init_masks:
                continue
            
            init_masks.append(mask)
        
        # Shuffle to remove order bias
        random.shuffle(init_masks)
        
        # Ensure we have exactly m distinct masks
        # (The generation above handles distinctness, but just in case)
        # If duplicates slipped in, replace them
        seen = set()
        final_init_masks = []
        for mask in init_masks:
            if mask not in seen:
                seen.add(mask)
                final_init_masks.append(mask)
            else:
                # Find a new unique mask
                while True:
                    size = random.randint(0, n)
                    mask = 0
                    elems = random.sample(range(n), size)
                    for e in elems:
                        mask |= (1 << e)
                    if mask not in seen:
                        seen.add(mask)
                        final_init_masks.append(mask)
                        break
        
        current_masks = final_init_masks
        current_score = _count_comparable_pairs_masks(current_masks)
        
        # 2. Simulated Annealing
        T = 20.0  # Initial temperature
        cooling = 0.9995  # Slow cooling
        min_T = 0.1
        
        local_best_score = current_score
        local_best_masks = list(current_masks)
        
        # Time budget per restart
        restart_start = time.time()
        restart_time_budget = 8.0  # 8 seconds per restart * 8 restarts = 64s total search + overhead
        
        while T > min_T:
            # Check global time
            if time.time() - start_time > max_time:
                return _masks_to_sets(best_overall_masks, n)
            
            # Check restart time
            if time.time() - restart_start > restart_time_budget:
                break
            
            # Pick a random index
            i = random.randint(0, m - 1)
            old_mask = current_masks[i]
            
            # Generate a new mask
            # Move 1: Add a random element not in mask
            # Move 2: Remove a random element from mask
            # Move 3: Swap an element
            # Move 4: Random mask biased towards current size
            
            move_type = random.random()
            
            if move_type < 0.3:
                # Add
                missing = [b for b in range(n) if not (old_mask & (1 << b))]
                if missing:
                    e = random.choice(missing)
                    new_mask = old_mask | (1 << e)
                else:
                    continue
            elif move_type < 0.6:
                # Remove
                present = [b for b in range(n) if (old_mask & (1 << b))]
                if present:
                    e = random.choice(present)
                    new_mask = old_mask & ~(1 << e)
                else:
                    continue
            elif move_type < 0.8:
                # Swap
                present = [b for b in range(n) if (old_mask & (1 << b))]
                missing = [b for b in range(n) if not (old_mask & (1 << b))]
                if present and missing:
                    e_in = random.choice(present)
                    e_out = random.choice(missing)
                    new_mask = (old_mask & ~(1 << e_in)) | (1 << e_out)
                else:
                    continue
            else:
                # Random neighbor: pick a random subset of a random neighbor or just random
                # Let's try: pick a random other set, and make new_mask a subset/superset of it
                j = random.randint(0, m - 1)
                if j == i:
                    j = (j + 1) % m
                neighbor_mask = current_masks[j]
                
                r = random.random()
                if r < 0.5:
                    # Subset of neighbor
                    # Remove random elements from neighbor
                    present = [b for b in range(n) if (neighbor_mask & (1 << b))]
                    k = random.randint(0, len(present))
                    to_remove = random.sample(present, k)
                    new_mask = neighbor_mask
                    for e in to_remove:
                        new_mask &= ~(1 << e)
                else:
                    # Superset of neighbor
                    # Add random elements to neighbor
                    missing = [b for b in range(n) if not (neighbor_mask & (1 << b))]
                    k = random.randint(0, len(missing))
                    to_add = random.sample(missing, k)
                    new_mask = neighbor_mask
                    for e in to_add:
                        new_mask |= (1 << e)
            
            if new_mask == old_mask:
                continue
            
            # Check distinctness
            is_dup = False
            for k in range(m):
                if k != i and current_masks[k] == new_mask:
                    is_dup = True
                    break
            
            if is_dup:
                continue
            
            # Calculate delta
            # Delta = (comparability of new_mask with others) - (comparability of old_mask with others)
            delta = 0
            for k in range(m):
                if k == i:
                    continue
                mk = current_masks[k]
                
                # Old comparison
                if old_mask != mk:
                    if (old_mask & mk) == old_mask:
                        delta -= 1
                    elif (old_mask & mk) == mk:
                        delta -= 1
                
                # New comparison
                if new_mask != mk:
                    if (new_mask & mk) == new_mask:
                        delta += 1
                    elif (new_mask & mk) == mk:
                        delta += 1
            
            # Accept?
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                current_masks[i] = new_mask
                current_score += delta
                
                if current_score > local_best_score:
                    local_best_score = current_score
                    local_best_masks = list(current_masks)
            
            T *= cooling
            
        # End of restart
        if local_best_score > best_overall_score:
            best_overall_score = local_best_score
            best_overall_masks = local_best_masks
        
        # Optional: restart from local best for next iteration?
        # For diversity, better to start fresh with new permutation.
        
    # Return best found
    if not best_overall_masks:
        # Fallback: return first restart result or random
        # This shouldn't happen if restarts > 0
        best_overall_masks = [0] * m
        
    return _masks_to_sets(best_overall_masks, n)
# EVOLVE-BLOCK-END
