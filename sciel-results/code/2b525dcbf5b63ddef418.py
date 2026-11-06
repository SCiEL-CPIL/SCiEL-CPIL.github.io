import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin for 60s limit

    n = 16
    k = 4
    all_sets = list(combinations(range(n), k))
    num_sets = len(all_sets)
    
    # Precompute bitmasks
    masks = [0] * num_sets
    for idx, s in enumerate(all_sets):
        m = 0
        for x in s:
            m |= (1 << x)
        masks[idx] = m

    best_family = []
    best_size = 0
    
    # Try multiple random restarts
    # We can also try to bias the randomization
    for restart in range(20):
        if time.time() > deadline:
            break
            
        random.seed() # Use system random
        indices = list(range(num_sets))
        random.shuffle(indices)
        
        current_family = []
        current_indices = set()
        
        for idx in indices:
            if time.time() > deadline:
                break
            
            # Check if idx can be added
            valid = True
            m_idx = masks[idx]
            
            if len(current_indices) < 2:
                valid = True
            else:
                fam_list = list(current_indices)
                L = len(fam_list)
                for i in range(L):
                    ia = fam_list[i]
                    ma = masks[ia]
                    K = ma & m_idx
                    for j in range(i + 1, L):
                        ib = fam_list[j]
                        mb = masks[ib]
                        if (mb & m_idx) == K and (mb & ma) == K:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                current_family.append(all_sets[idx])
                current_indices.add(idx)
                
                if len(current_family) > best_size:
                    best_size = len(current_family)
                    best_family = current_family[:]
                    
                    if best_size >= 250:
                        return [sorted(s) for s in best_family]

    return [sorted(s) for s in best_family]
