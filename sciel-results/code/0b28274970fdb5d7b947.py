def construct():
    import itertools
    import random
    import numpy as np
    import time

    n = 16
    all_points = list(range(n))
    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)
    
    # Precompute masks for triples
    triple_masks = []
    for t in all_triples:
        m = 0
        for p in t:
            m |= (1 << p)
        triple_masks.append(m)
        
    # Precompute popcounts for all masks
    max_mask = 1 << n
    popcounts = np.zeros(max_mask, dtype=np.int8)
    for i in range(1, max_mask):
        popcounts[i] = popcounts[i >> 1] + (i & 1)
        
    all_masks_np = np.arange(max_mask, dtype=np.int32)

    def calc_alpha(triple_indices):
        """
        Calculate independence number for a set of triple indices.
        """
        if not triple_indices:
            return n
            
        is_indep = np.ones(max_mask, dtype=bool)
        
        for idx in triple_indices:
            tm = triple_masks[idx]
            is_indep[(all_masks_np & tm) == tm] = False
            
        return int(np.max(popcounts[is_indep]))

    best_alpha = n + 1
    best_indices = []
    
    start_time = time.time()
    time_limit = 115
    
    # Phase 1: Random Greedy
    # We can increase attempts significantly due to fast alpha calc
    max_attempts = 5000
    
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            break
            
        random.seed(attempt)
        shuffled_indices = list(range(num_triples))
        random.shuffle(shuffled_indices)
        
        cur_indices = []
        used_pairs = set()
        
        for idx in shuffled_indices:
            t = all_triples[idx]
            a, b, c = t
            # Normalize pairs
            p1 = (a, b) if a < b else (b, a)
            p2 = (a, c) if a < c else (c, a)
            p3 = (b, c) if b < c else (c, b)
            
            if p1 in used_pairs or p2 in used_pairs or p3 in used_pairs:
                continue
                
            cur_indices.append(idx)
            used_pairs.add(p1)
            used_pairs.add(p2)
            used_pairs.add(p3)
            
        alpha = calc_alpha(cur_indices)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_indices = cur_indices[:]
            
            if best_alpha <= 7:
                break

    # Phase 2: Local Search (Optional, if time remains)
    if time.time() - start_time < time_limit - 2 and best_indices:
        cur_indices = best_indices[:]
        cur_alpha = best_alpha
        used_pairs = set()
        for idx in cur_indices:
            t = all_triples[idx]
            a, b, c = t
            p1 = (a, b) if a < b else (b, a)
            p2 = (a, c) if a < c else (c, a)
            p3 = (b, c) if b < c else (c, b)
            used_pairs.add(p1)
            used_pairs.add(p2)
            used_pairs.add(p3)
            
        local_start = time.time()
        while time.time() - local_start < 10:
            # Try a random swap
            if not cur_indices:
                break
            remove_idx_pos = random.randint(0, len(cur_indices) - 1)
            remove_idx = cur_indices[remove_idx_pos]
            
            # Remove
            t_rem = all_triples[remove_idx]
            a, b, c = t_rem
            p1 = (a, b) if a < b else (b, a)
            p2 = (a, c) if a < c else (c, a)
            p3 = (b, c) if b < c else (c, b)
            used_pairs.discard(p1)
            used_pairs.discard(p2)
            used_pairs.discard(p3)
            cur_indices.pop(remove_idx_pos)
            
            # Try to add a random valid triple
            added = False
            for _ in range(10): # Try a few random additions
                add_idx = random.randint(0, num_triples - 1)
                if add_idx in cur_indices: # Inefficient check, but n is small? 
                    # cur_indices is list. 'in' is O(N). N~30. OK.
                    continue
                t_add = all_triples[add_idx]
                a2, b2, c2 = t_add
                q1 = (a2, b2) if a2 < b2 else (b2, a2)
                q2 = (a2, c2) if a2 < c2 else (c2, a2)
                q3 = (b2, c2) if b2 < c2 else (c2, b2)
                
                if q1 in used_pairs or q2 in used_pairs or q3 in used_pairs:
                    continue
                    
                # Valid
                cur_indices.append(add_idx)
                used_pairs.add(q1)
                used_pairs.add(q2)
                used_pairs.add(q3)
                added = True
                break
                
            if not added:
                # Revert removal
                cur_indices.insert(remove_idx_pos, remove_idx)
                used_pairs.add(p1)
                used_pairs.add(p2)
                used_pairs.add(p3)
                continue
                
            new_alpha = calc_alpha(cur_indices)
            
            if new_alpha < cur_alpha:
                cur_alpha = new_alpha
                if cur_alpha < best_alpha:
                    best_alpha = cur_alpha
                    best_indices = cur_indices[:]
                    if best_alpha <= 7:
                        break
            else:
                # Revert addition and removal
                cur_indices.pop()
                used_pairs.discard(q1)
                used_pairs.discard(q2)
                used_pairs.discard(q3)
                cur_indices.insert(remove_idx_pos, remove_idx)
                used_pairs.add(p1)
                used_pairs.add(p2)
                used_pairs.add(p3)

    return [list(all_triples[i]) for i in best_indices]
