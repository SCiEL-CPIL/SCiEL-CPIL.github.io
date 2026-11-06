import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    all_points = list(range(16))
    all_triples_list = list(itertools.combinations(all_points, 3))
    # Map triple tuple to mask
    triple_masks = {}
    for t in all_triples_list:
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks[t] = mask
        
    num_triples = len(all_triples_list)
    
    # Precompute conflicts: conflict[i] is a set of indices j that conflict with i
    # Using list of sets for fast lookup? Or bitsets?
    # 560 triples. Bitset is 560 bits. Python int is good.
    conflict_bits = [0] * num_triples
    for i in range(num_triples):
        mi = triple_masks[all_triples_list[i]]
        for j in range(i+1, num_triples):
            mj = triple_masks[all_triples_list[j]]
            inter = mi & mj
            if inter & (inter - 1): # share > 1 point
                conflict_bits[i] |= (1 << j)
                conflict_bits[j] |= (1 << i)

    def is_valid_add(active_mask, idx):
        # active_mask is a bitset of active triple indices
        return (conflict_bits[idx] & active_mask) == 0

    def calculate_alpha(active_indices):
        # active_indices is a set of indices
        if not active_indices:
            return 16, None
            
        # Precompute masks of active triples
        active_masks = [triple_masks[all_triples_list[i]] for i in active_indices]
        
        # Check subsets from largest to smallest
        # We can optimize by checking if a subset is independent
        # A subset S is independent if no active mask is a subset of S
        
        # To find the max size, we can iterate k downwards.
        # For efficiency, we can just check all subsets of size k.
        
        for k in range(16, 0, -1):
            found_independent = False
            # Iterate all subsets of size k
            for subset in itertools.combinations(all_points, k):
                smask = 0
                for p in subset:
                    smask |= (1 << p)
                
                is_ind = True
                for tm in active_masks:
                    if (tm & smask) == tm:
                        is_ind = False
                        break
                if is_ind:
                    return k, subset
            # If no independent set of size k found, continue to k-1
            
        return 0, None

    def generate_initial():
        active = set()
        indices = list(range(num_triples))
        random.shuffle(indices)
        for idx in indices:
            if is_valid_add(0, idx): # Wait, active is a set, not a bitmask for this check
                # Check against active set
                valid = True
                for j in active:
                    if conflict_bits[idx] & (1 << j):
                        valid = False
                        break
                if valid:
                    active.add(idx)
        return active

    # Better initial generation using bitmask for speed
    def generate_initial_fast():
        active_mask = 0
        indices = list(range(num_triples))
        random.shuffle(indices)
        for idx in indices:
            if (conflict_bits[idx] & active_mask) == 0:
                active_mask |= (1 << idx)
        # Convert to set for easier manipulation in local search?
        # Or keep as bitmask?
        # Bitmask is faster for conflict checks.
        return active_mask

    def mask_to_set(mask):
        return set(i for i in range(num_triples) if (mask >> i) & 1)

    def set_to_mask(s):
        m = 0
        for i in s:
            m |= (1 << i)
        return m

    best_alpha = 16
    best_mask = generate_initial_fast()
    best_alpha, _ = calculate_alpha(mask_to_set(best_mask))
    
    # Simulated Annealing
    temperature = 1.0
    cooling = 0.999
    
    current_mask = best_mask
    current_alpha, current_ind_set = calculate_alpha(mask_to_set(current_mask))
    
    while time.time() < deadline:
        if current_alpha <= 6: # Theoretical lower bound is likely 6 or 7
             # If we hit 6, we can try to get 5, but 6 is very good.
             # Let's keep going to see if we can hit 5.
             pass
        
        # Find a random independent set of current_alpha size
        # calculate_alpha returns one. Let's get one.
        if current_alpha > 0:
             # We need an independent set to target.
             # The previous call returned one. Let's store it.
             pass
        else:
             current_alpha, current_ind_set = calculate_alpha(mask_to_set(current_mask))

        if current_ind_set is None:
            break
            
        # Pick a random triple from the independent set
        # To reduce alpha, we need to add a triple that intersects every independent set of size current_alpha.
        # Adding a triple from the current independent set breaks it.
        
        triples_in_ind = list(itertools.combinations(current_ind_set, 3))
        if not triples_in_ind:
             break
             
        target_triple = random.choice(triples_in_ind)
        target_idx = all_triples_list.index(target_triple) # Slow!
        
        # Precompute index lookup
        # Let's create a dict for triple -> index
        # We did this? No, we have triple_masks dict.
        # Let's make a reverse dict.
        
        # ... (Need to fix index lookup)
        
        # Let's restructure to use indices directly.
        
        # ...
        
        # For now, assume we have target_idx.
        
        # Calculate conflicts
        conflicts = conflict_bits[target_idx] & current_mask
        # Remove conflicts
        new_mask = current_mask & ~conflicts
        # Add target
        new_mask |= (1 << target_idx)
        
        # Refill: try to add random triples
        # This is expensive if done fully.
        # Let's do a few random additions.
        indices = list(range(num_triples))
        random.shuffle(indices)
        for idx in indices:
             if (new_mask >> idx) & 1:
                 continue
             if (conflict_bits[idx] & new_mask) == 0:
                 new_mask |= (1 << idx)
                 # Stop after some additions to keep it fast?
                 # Or full refill? Full refill is better for coverage.
        
        # Evaluate new alpha
        new_alpha, new_ind_set = calculate_alpha(mask_to_set(new_mask))
        
        # Acceptance
        if new_alpha < current_alpha:
            current_alpha = new_alpha
            current_mask = new_mask
            current_ind_set = new_ind_set
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_mask = current_mask
        elif new_alpha == current_alpha:
            # Accept if density increased?
            # Density = popcount(mask)
            if bin(new_mask).count('1') > bin(current_mask).count('1'):
                 current_mask = new_mask
                 current_alpha, current_ind_set = calculate_alpha(mask_to_set(current_mask))
            elif random.random() < 0.1:
                 current_mask = new_mask
                 current_alpha, current_ind_set = calculate_alpha(mask_to_set(current_mask))
        else:
            # Reject or accept with small prob
            if random.random() < 0.01:
                 current_mask = new_mask
                 current_alpha, current_ind_set = calculate_alpha(mask_to_set(current_mask))
                 
        temperature *= cooling

    return [list(all_triples_list[i]) for i in mask_to_set(best_mask)]
