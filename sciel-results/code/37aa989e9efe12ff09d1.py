import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    # Precompute all 4-subsets of [1,16] as bitmasks
    all_sets = []
    for combo in combinations(range(1, 17), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append((mask, combo))
    
    n_sets = len(all_sets)
    
    # Precompute pairwise intersection sizes
    # inter[i][j] = popcount(mask_i & mask_j)
    # This is n_sets^2 which is ~3.3M, feasible
    masks = [m for m, _ in all_sets]
    
    # Use a greedy approach with conflict tracking
    # For each candidate set, check if adding it creates a sunflower with any two existing sets
    # A sunflower exists if there exist a, b in chosen such that:
    #   |a∩b| = |a∩new| = |b∩new|
    
    # To speed up: for each pair of chosen sets (a,b), we know their intersection size s = |a∩b|
    # We need to avoid adding a set c such that |a∩c| = s and |b∩c| = s
    
    # Greedy with randomization and multiple restarts
    import random
    random.seed(42)
    
    best_family = []
    
    # Try multiple orderings
    for trial in range(50):
        if time.time() > deadline:
            break
            
        # Shuffle the order of consideration
        indices = list(range(n_sets))
        random.shuffle(indices)
        
        chosen = []
        chosen_masks = []
        # For fast lookup: map from mask to index in chosen
        mask_to_idx = {}
        
        # To check conflicts efficiently:
        # When considering adding set c with mask mc:
        # For each pair (a, b) in chosen, if |a∩b| = s, we need |a∩c| != s or |b∩c| != s
        # This is O(k^2) per candidate where k = len(chosen)
        
        # Better: maintain for each intersection size s, the pairs that have that size
        # But checking all pairs is still O(k^2)
        
        # Alternative: for each existing set a, precompute what intersection sizes it has with c
        # Still O(k) per candidate, but we need to check all pairs...
        
        # Let's just do the O(k^2) check but break early
        for idx in indices:
            if time.time() > deadline:
                break
                
            mask = masks[idx]
            combo = all_sets[idx][1]
            
            # Check if adding this creates a sunflower
            conflict = False
            k = len(chosen_masks)
            
            # For each pair in chosen, check if intersection with new set matches
            for i in range(k):
                mi = chosen_masks[i]
                inter_ic = bin(mi & mask).count('1')
                
                for j in range(i + 1, k):
                    mj = chosen_masks[j]
                    inter_ij = bin(mi & mj).count('1')
                    inter_jc = bin(mj & mask).count('1')
                    
                    if inter_ij == inter_ic == inter_jc:
                        conflict = True
                        break
                if conflict:
                    break
            
            if not conflict:
                chosen_masks.append(mask)
                chosen.append(combo)
        
        if len(chosen) > len(best_family):
            best_family = chosen[:]
        
        # Early stop if we can't beat best
        # Theoretical max is bounded, but let's just run trials
    
    return [sorted(s) for s in best_family]
