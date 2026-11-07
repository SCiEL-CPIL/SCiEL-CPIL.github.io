import time

def construct():
    n_elements = 12
    universe_size = 1 << n_elements
    family_size = 40
    
    # Precompute all masks
    all_masks = list(range(universe_size))
    
    # We will greedily select masks
    family = []
    used = set()
    
    # To make it deterministic and potentially better, we can sort candidates initially
    # or just iterate.
    # To speed up, we can pre-compute bit counts or just rely on bitwise ops.
    
    for _ in range(family_size):
        best_mask = -1
        best_score = -1
        
        for m in all_masks:
            if m in used:
                continue
            
            # Count comparabilities with current family
            score = 0
            for s in family:
                if (m & s) == m or (m & s) == s:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_mask = m
                
        if best_mask != -1:
            family.append(best_mask)
            used.add(best_mask)
        else:
            # Fallback (should not happen for 40 < 4096)
            for m in all_masks:
                if m not in used:
                    family.append(m)
                    used.add(m)
                    break

    # Convert masks to lists of integers
    result = []
    for mask in family:
        subset = []
        for i in range(n_elements):
            if mask & (1 << i):
                subset.append(i)
        result.append(subset)
        
    return result
