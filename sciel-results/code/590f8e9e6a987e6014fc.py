"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 12, 'm': 40}

def _count_comparable_pairs(sets):
    """
    Count the number of unordered pairs (i, j) with i < j such that 
    sets[i] is a strict subset of sets[j] or sets[j] is a strict subset of sets[i].
    Uses bitmask representation for efficiency.
    """
    n = P['n']
    m = len(sets)
    masks = [0] * m
    for i in range(m):
        mask = 0
        for v in sets[i]:
            if 0 <= v < n:
                mask |= (1 << v)
        masks[i] = mask
    
    count = 0
    for i in range(m):
        mi = masks[i]
        for j in range(i + 1, m):
            mj = masks[j]
            if mi == mj:
                continue
            inter = mi & mj
            if inter == mi or inter == mj:
                count += 1
    return count

def _random_subset_of_size(k, n):
    """Generate a random subset of size k from 0..n-1."""
    if k == 0:
        return []
    if k >= n:
        return list(range(n))
    return random.sample(range(n), k)

def construct():
    n = P['n']
    m = P['m']
    
    start_time = time.time()
    max_time = 85.0
    
    # Strategy: Build a dense comparability structure using a "spine" chain.
    # A chain of sets C_0 ⊂ C_1 ⊂ ... ⊂ C_12 contains 13 sets.
    # Any subset of a spine set is comparable with that spine set (as a subset).
    # Any superset of a spine set is comparable with that spine set (as a superset).
    
    # Initialization:
    # 1. Create a random permutation of the 12 elements to define the spine.
    perm = list(range(n))
    random.shuffle(perm)
    
    # Spine sets: S_k = {perm[0], ..., perm[k-1]} for k=0..12
    # S_0 = [], S_1 = {perm[0]}, ..., S_12 = {0..11}
    spine_masks = [0] * (n + 1)
    current_mask = 0
    for k in range(n + 1):
        spine_masks[k] = current_mask
        if k < n:
            current_mask |= (1 << perm[k])
            
    # We have 13 spine sets. We need 40 total.
    # The remaining 27 sets should be chosen to maximize comparability with the spine and each other.
    # Ideally, we want sets that are subsets of some spine sets or supersets of some spine sets.
    
    # Let's start with the 13 spine sets.
    sets_list = [0] * (n + 1)
    for k in range(n + 1):
        sets_list[k] = spine_masks[k]
        
    # Now fill the remaining m - (n+1) slots.
    # We can add subsets of the full set that are "close" to the spine.
    # Actually, any subset of the universe is comparable with the spine sets that contain it or are contained in it.
    # Let's just generate random sets and use SA to optimize.
    
    current_sets = [list(s) for s in [[v for v in range(n) if (mask >> v) & 1] for mask in sets_list]]
    
    # Add random unique sets to reach size m
    existing_masks = set(sets_list)
    for _ in range(m - len(current_sets)):
        while True:
            size = random.randint(0, n)
            s = _random_subset_of_size(size, n)
            mask = 0
            for v in s:
                mask |= (1 << v)
            if mask not in existing_masks:
                existing_masks.add(mask)
                current_sets.append(s)
                break
    
    current_score = _count_comparable_pairs(current_sets)
    best_sets = [list(s) for s in current_sets]
    best_score = current_score
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.995
    min_T = 0.5
    
    while T > min_T and time.time() - start_time < max_time:
        idx = random.randint(0, m - 1)
        old_s = current_sets[idx]
        old_mask = 0
        for v in old_s:
            old_mask |= (1 << v)
        
        # Generate a new candidate set
        # Bias towards sets that are subsets or supersets of existing sets to increase comparability
        r = random.random()
        if r < 0.6:
            # Modify by adding or removing one element
            if old_s:
                # Remove a random element
                idx_remove = random.randint(0, len(old_s) - 1)
                new_s = old_s[:idx_remove] + old_s[idx_remove+1:]
            else:
                # Add a random element
                missing = [v for v in range(n) if v not in old_s]
                if missing:
                    new_s = old_s + [random.choice(missing)]
                else:
                    continue
        else:
            # Swap an element
            if not old_s:
                missing = [v for v in range(n)]
                if missing:
                    new_s = [random.choice(missing)]
                else:
                    continue
            else:
                idx_remove = random.randint(0, len(old_s) - 1)
                e_remove = old_s[idx_remove]
                missing = [v for v in range(n) if v not in old_s]
                if missing:
                    e_add = random.choice(missing)
                    new_s = old_s[:idx_remove] + [e_add] + old_s[idx_remove+1:]
                else:
                    continue
        
        new_mask = 0
        for v in new_s:
            new_mask |= (1 << v)
            
        if new_mask == old_mask:
            continue
            
        # Check distinctness
        if new_mask in existing_masks and new_mask != old_mask:
            # It's a duplicate of another set
            continue
            
        # Update existing masks
        existing_masks.remove(old_mask)
        existing_masks.add(new_mask)
        
        current_sets[idx] = new_s
        new_score = _count_comparable_pairs(current_sets)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < pow(2.718281828, delta / T):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_sets = [list(s) for s in current_sets]
        else:
            # Revert
            current_sets[idx] = old_s
            existing_masks.remove(new_mask)
            existing_masks.add(old_mask)
            
        T *= cooling
        
    # Final validation and type conversion
    result = []
    seen = set()
    for s in best_sets:
        t = tuple(sorted(s))
        if t not in seen:
            seen.add(t)
            result.append([int(x) for x in s])
        else:
            # Should not happen, but if it does, find a replacement
            # This is a fallback, should rarely be triggered
            for mask in range(1 << n):
                s_new = [b for b in range(n) if (mask >> b) & 1]
                t_new = tuple(sorted(s_new))
                if t_new not in seen:
                    seen.add(t_new)
                    result.append(s_new)
                    break
            
    return result
# EVOLVE-BLOCK-END
