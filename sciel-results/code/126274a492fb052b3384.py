import random
import time

P = {'n': 12, 'm': 40}

def construct():
    n = P['n']
    m = P['m']
    
    # 1. Initialize with the "core" of the lattice: Empty set and all singletons.
    # These are highly connected.
    family = set()
    family.add(0) # Empty set
    
    # Add all singletons
    for i in range(n):
        family.add(1 << i)
        
    current_size = len(family) # 13
    target_size = m # 40
    
    # Precompute all masks grouped by size to limit search space
    # We likely won't need sets of size > 6 for m=40
    max_size_to_consider = 6
    candidates_by_size = {s: [] for s in range(2, max_size_to_consider + 1)}
    
    for mask in range(1, 1 << n):
        size = bin(mask).count('1')
        if 2 <= size <= max_size_to_consider:
            candidates_by_size[size].append(mask)
            
    # Combine all candidates
    all_candidates = []
    for s in range(2, max_size_to_consider + 1):
        all_candidates.extend(candidates_by_size[s])
        
    # Greedy Selection
    # We need to add (target_size - current_size) sets
    # For each step, pick the candidate that maximizes comparable pairs with existing family
    
    while current_size < target_size:
        best_candidate = None
        best_gain = -1
        
        for c in all_candidates:
            if c in family:
                continue
                
            gain = 0
            # Check comparability with all existing members
            for f in family:
                if (c & f) == c: # c is subset of f
                    gain += 1
                elif (c & f) == f: # f is subset of c
                    gain += 1
                    
            if gain > best_gain:
                best_gain = gain
                best_candidate = c
                
        if best_candidate is not None:
            family.add(best_candidate)
            current_size += 1
        else:
            # Fallback if no candidates left (should not happen)
            break
            
    # Optional: Local Search to polish
    # Convert to list for indexing
    fam_list = list(family)
    
    def count_comparable(fam):
        count = 0
        for i in range(len(fam)):
            for j in range(i+1, len(fam)):
                a = fam[i]
                b = fam[j]
                if (a & b) == a or (a & b) == b:
                    count += 1
        return count
        
    current_score = count_comparable(fam_list)
    
    # Simple Local Search: Try to swap one element with another to improve score
    # Time limit 40s
    start_time = time.time()
    time_limit = 40.0
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() - start_time < time_limit and no_improve_count < max_no_improve:
        improved = False
        # Pick a random element to replace
        i = random.randint(0, len(fam_list) - 1)
        old_mask = fam_list[i]
        
        # Generate a random neighbor (flip 1-2 bits)
        new_mask = old_mask
        num_flips = random.randint(1, 2)
        for _ in range(num_flips):
            bit = random.randint(0, n - 1)
            new_mask ^= (1 << bit)
            
        if new_mask in family:
            continue
            
        # Calculate delta
        old_contrib = 0
        new_contrib = 0
        for j, f in enumerate(fam_list):
            if j == i:
                continue
            if (old_mask & f) == old_mask or (old_mask & f) == f:
                old_contrib += 1
            if (new_mask & f) == new_mask or (new_mask & f) == f:
                new_contrib += 1
                
        delta = new_contrib - old_contrib
        
        if delta > 0:
            # Accept
            family.remove(old_mask)
            family.add(new_mask)
            fam_list[i] = new_mask
            current_score += delta
            improved = True
            no_improve_count = 0
        else:
            no_improve_count += 1
            if not improved:
                # Cool down or just continue? 
                # Since we only accept improvements, this is a hill climb.
                # If we get stuck, we might need SA, but hill climb is fast.
                pass

    # Convert to lists of ints
    result = []
    for mask in family:
        s = []
        for i in range(n):
            if mask & (1 << i):
                s.append(i)
        result.append(s)
        
    return result
