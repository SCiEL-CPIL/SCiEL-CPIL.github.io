import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Generate all 4-subsets of [16] as bitmasks
    all_masks = []
    for combo in combinations(range(16), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_masks.append(mask)
    
    masks = np.array(all_masks, dtype=np.uint32)
    n_sets = len(masks)
    
    def mask_to_list(mask):
        s = []
        for i in range(16):
            if mask & (1 << i):
                s.append(i + 1)
        return sorted(s)
    
    best_family_size = 0
    best_family = []
    
    def can_add(S, family):
        m = len(family)
        if m < 2:
            return True
        # Early exit: if family is large, this will be slow
        for i in range(m):
            Ai = family[i]
            inter_iS = Ai & S
            for j in range(i + 1, m):
                Bj = family[j]
                if (Ai & Bj) == inter_iS and (Bj & S) == inter_iS:
                    return False
        return True
    
    for attempt in range(40):
        if time.time() > deadline:
            break
        
        seed = attempt * 7919 + 1
        rng = random.Random(seed)
        indices = list(range(n_sets))
        rng.shuffle(indices)
        
        family = []
        
        for idx in indices:
            if time.time() > deadline:
                break
            S = masks[idx]
            if can_add(S, family):
                family.append(S)
        
        current_size = len(family)
        if current_size > best_family_size:
            best_family_size = current_size
            best_family = list(family)
        
        # Local search
        if time.time() < deadline - 8:
            improved = True
            while improved and time.time() < deadline - 3:
                improved = False
                for _ in range(50):
                    if time.time() > deadline:
                        break
                    idx = rng.randint(0, n_sets - 1)
                    S = masks[idx]
                    if S not in family and can_add(S, family):
                        family.append(S)
                        improved = True
                        break
    
    result = []
    for mask in best_family:
        result.append(mask_to_list(mask))
    
    return result
