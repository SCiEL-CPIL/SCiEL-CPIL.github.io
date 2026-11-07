"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    
    # Helper to check if two masks are comparable
    def is_comparable(a, b):
        return (a & b) == a or (a & b) == b

    # Helper to count comparable pairs with a list of masks
    def count_comparable_with(mask, masks):
        count = 0
        for s in masks:
            if is_comparable(mask, s):
                count += 1
        return count

    # 1. Greedy Initialization
    selected = []
    available = set(range(total_masks))
    
    for _ in range(m):
        best_score = -1
        best_mask = -1
        
        for mask in available:
            score = count_comparable_with(mask, selected)
            if score > best_score:
                best_score = score
                best_mask = mask
        
        selected.append(best_mask)
        available.remove(best_mask)
        
    # 2. Local Search Improvement
    unselected = available
    
    # Pre-compute comparability with current selected sets?
    # We will just recalculate in the loop, it's fast enough.
    
    improved = True
    while improved:
        improved = False
        best_gain = 0
        best_i = -1
        best_u = -1
        
        # We iterate over all pairs (selected, unselected) to find the best swap
        for i, s_i in enumerate(selected):
            # Calculate current contribution of s_i to the total score (excluding itself)
            # This is count_comparable_with(s_i, selected without s_i)
            # But we can just calculate the gain directly.
            # Gain = C(u, S \ {s_i}) - C(s_i, S \ {s_i})
            
            # To avoid O(M) inside the O(N) loop, we can precompute C(s_i, S \ {s_i})
            # But M is 40, N is 4096. 40*4096 = 160k. Inner loop 40. Total 6.4M.
            
            for u in unselected:
                # Calculate C(u, S \ {s_i})
                # We can iterate through selected and skip i
                count_u = 0
                for j, s_j in enumerate(selected):
                    if j == i:
                        continue
                    if is_comparable(u, s_j):
                        count_u += 1
                
                count_si = 0
                for j, s_j in enumerate(selected):
                    if j == i:
                        continue
                    if is_comparable(s_i, s_j):
                        count_si += 1
                        
                gain = count_u - count_si
                
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_u = u
                    
        if best_gain > 0:
            # Perform swap
            s_old = selected[best_i]
            selected[best_i] = best_u
            unselected.remove(best_u)
            unselected.add(s_old)
            improved = True

    # Convert to lists
    result = []
    for mask in selected:
        result.append([b for b in range(n) if (mask >> b) & 1])
        
    return result

# EVOLVE-BLOCK-END
