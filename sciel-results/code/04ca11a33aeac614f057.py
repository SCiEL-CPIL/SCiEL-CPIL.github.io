import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s margin for safety
    
    # Generate all 4-subsets of [1, 16] as bitmasks
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for i in combo:
            mask |= (1 << i)
        all_sets.append(mask)
    
    n_sets = len(all_sets)
    all_mask = (1 << 16) - 1
    
    # Map mask to index
    mask_to_idx = {m: i for i, m in enumerate(all_sets)}
    
    best_size = 0
    best_family = []
    
    # Helper to get list of submasks of 'mask' of size 'size'
    def get_submasks(mask, size):
        if size == 0:
            return [0]
        indices = []
        for i in range(16):
            if mask & (1 << i):
                indices.append(i)
        if len(indices) < size:
            return []
        res = []
        for c in combinations(indices, size):
            m = 0
            for i in c:
                m |= (1 << i)
            res.append(m)
        return res

    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Shuffle indices
        order = list(range(n_sets))
        random.shuffle(order)
        
        family_indices = []
        blocked = [0] * n_sets
        
        for idx_A in order:
            if time.time() > deadline:
                break
                
            if blocked[idx_A] > 0:
                continue
            
            # Add A to family
            family_indices.append(idx_A)
            A = all_sets[idx_A]
            
            # Update blocked for all pairs (A, B) where B is in family (excluding A)
            # We iterate over existing family members
            for i in range(len(family_indices) - 1):
                idx_B = family_indices[i]
                B = all_sets[idx_B]
                
                # Compute W(A, B)
                K = A & B
                k_size = K.bit_count()
                rem = 4 - k_size
                
                if rem < 0:
                    continue # Should not happen for distinct 4-sets
                
                # forbidden part: elements in A or B but not in K
                forb = (A | B) & ~K
                avail = all_mask & ~forb
                
                # Generate submasks of avail of size rem
                submasks = get_submasks(avail, rem)
                
                for S_prime in submasks:
                    S = S_prime | K
                    # S is a 4-set?
                    # Yes, by construction.
                    # Check if S is in all_sets
                    # It should be, since it's a 4-subset of [16]
                    idx_S = mask_to_idx[S]
                    blocked[idx_S] += 1
        
        if len(family_indices) > best_size:
            best_size = len(family_indices)
            best_family = family_indices[:]
            
            if best_size >= 250:
                break

    # Convert best_family back to list of sorted lists of integers 1..16
    result = []
    for idx in best_family:
        mask = all_sets[idx]
        s = []
        for i in range(16):
            if mask & (1 << i):
                s.append(i + 1)
        result.append(sorted(s))
        
    return result
