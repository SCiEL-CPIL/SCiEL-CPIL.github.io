import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    n = 12
    m = 40
    
    # Base sets
    base_sets = []
    base_sets.append([]) # Empty
    for i in range(n):
        base_sets.append([i]) # Singletons
    base_sets.append(list(range(n))) # Full
    for i in range(n):
        s = [x for x in range(n) if x != i]
        base_sets.append(s) # Size 11
    
    current_sets = base_sets
    current_masks = set(frozenset(s) for s in current_sets)
    
    # Candidate pool: Sizes 9, 10, 8, ...
    # We will generate candidates on the fly or pre-generate the most likely ones.
    # Sizes 10 and 9 are most likely to be added.
    # Let's generate all sets of size 9 and 10.
    candidates = []
    for r in [10, 9, 8, 7]:
        for combo in itertools.combinations(range(n), r):
            s = list(combo)
            fs = frozenset(s)
            if fs not in current_masks:
                candidates.append(s)
        
        # Check time
        if time.time() > deadline:
            break

    # If we need more candidates (if 9,10,8,7 isn't enough? 66+220+495+792 > 14)
    # We have plenty.
    
    # Greedy Selection
    while len(current_sets) < m:
        if time.time() > deadline:
            break
            
        best_gain = -1
        best_set = None
        
        for c in candidates:
            c_fs = frozenset(c)
            if c_fs in current_masks:
                continue
                
            gain = 0
            # Check subsets of c in current_sets
            # Check supersets of c in current_sets
            for s in current_sets:
                s_fs = frozenset(s)
                if s_fs == c_fs:
                    continue
                if s_fs < c_fs:
                    gain += 1
                elif c_fs < s_fs:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_set = c
                
        if best_set is None:
            break
            
        current_sets.append(best_set)
        current_masks.add(frozenset(best_set))
        
        # Remove from candidates to avoid re-checking? 
        # Or just let the 'if c_fs in current_masks' handle it.
        # For efficiency, we could remove it, but list removal is O(N).
        # Given small N, it's fine.
        
    # Final check and cleanup
    # Ensure exactly 40 distinct lists
    # If we didn't reach 40, fill with random distinct sets?
    # But greedy should reach it.
    
    if len(current_sets) < m:
        # Fallback: add random sets
        import random
        existing = set(frozenset(s) for s in current_sets)
        while len(current_sets) < m:
            r = random.randint(0, n-1)
            s = list(random.sample(range(n), r))
            fs = frozenset(s)
            if fs not in existing:
                current_sets.append(s)
                existing.add(fs)
            if time.time() > deadline:
                break

    # Return list of lists
    return [list(s) for s in current_sets]
