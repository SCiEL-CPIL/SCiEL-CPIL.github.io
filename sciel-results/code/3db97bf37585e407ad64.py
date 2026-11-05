# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Precompute all 4-subsets as bitmasks
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for i in combo:
            mask |= (1 << i)
        all_sets.append(mask)
    
    n_sets = len(all_sets)
    best_family = []
    best_size = 0
    
    # Try multiple random orderings
    max_tries = 50
    for trial in range(max_tries):
        if time.time() > deadline:
            break
        
        # Create a random permutation
        indices = list(range(n_sets))
        random.shuffle(indices)
        
        chosen = []
        chosen_masks = set()
        
        for idx in indices:
            if time.time() > deadline:
                break
            
            mask = all_sets[idx]
            if mask in chosen_masks:
                continue
            
            # Check against all pairs in chosen
            valid = True
            for i in range(len(chosen)):
                for j in range(i + 1, len(chosen)):
                    a = chosen[i]
                    b = chosen[j]
                    # Check if a, b, mask form a sunflower
                    ab = a & b
                    am = a & mask
                    bm = b & mask
                    if ab == am == bm:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                chosen.append(mask)
                chosen_masks.add(mask)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_family = chosen[:]
    
    # Convert bitmasks back to sorted lists
    result = []
    for mask in best_family:
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
    
    return result
# EVOLVE-BLOCK-END
