import random
import itertools
import time

def construct():
    elements = list(range(1, 17))
    all_sets = list(itertools.combinations(elements, 4))
    n_sets = len(all_sets)
    set_to_idx = {s: i for i, s in enumerate(all_sets)}
    
    best_family = []
    start_time = time.time()
    time_limit = 85.0
    
    # Precompute sets as bitmasks?
    # Might be faster for intersection/union.
    # 16 elements -> 16 bits.
    masks = [0] * n_sets
    for i, s in enumerate(all_sets):
        m = 0
        for x in s:
            m |= (1 << (x - 1))
        masks[i] = m
        
    # Precompute popcount?
    # Python 3.8+ has int.bit_count()
    
    for restart in range(500):
        if time.time() - start_time > time_limit:
            break
            
        order = list(range(n_sets))
        random.shuffle(order)
        
        blocked = [False] * n_sets
        family = []
        
        for idx in order:
            if blocked[idx]:
                continue
                
            family.append(idx)
            
            # Update blocked
            mask_s = masks[idx]
            
            for j in range(len(family) - 1):
                b_idx = family[j]
                mask_b = masks[b_idx]
                
                # K = mask_s & mask_b
                K_mask = mask_s & mask_b
                K_len = K_mask.bit_count()
                
                # Union
                U_mask = mask_s | mask_b
                
                # V = [1..16] \ U_mask
                # We need to find combinations of V of size 4 - K_len
                # V is represented by bits not in U_mask.
                # We can iterate through all 1820 sets and check?
                # No, that's 1820 * 1000 * 1000.
                
                # Better: extract elements of V.
                # V elements are x in 1..16 such that bit x-1 is 0 in U_mask.
                V_elems = []
                for x in range(1, 17):
                    if not (U_mask >> (x - 1)) & 1:
                        V_elems.append(x)
                        
                need = 4 - K_len
                if need > len(V_elems):
                    continue
                
                for comb in itertools.combinations(V_elems, need):
                    # S = K | comb
                    # K elements:
                    K_elems = []
                    for x in range(1, 17):
                        if (K_mask >> (x - 1)) & 1:
                            K_elems.append(x)
                    
                    S = tuple(sorted(K_elems + list(comb)))
                    s_idx = set_to_idx[S]
                    blocked[s_idx] = True

        if len(family) > len(best_family):
            best_family = family

    return [list(all_sets[i]) for i in best_family]
