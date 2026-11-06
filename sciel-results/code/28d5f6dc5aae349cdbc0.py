def construct():
    import itertools
    import random
    import numpy as np
    import time

    n = 16
    all_points = list(range(n))
    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)
    
    # Map triple tuple to index
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
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

    def calc_alpha_witness(triple_indices):
        """
        Calculate independence number and return (alpha, witness_mask).
        """
        if not triple_indices:
            return n, (1 << n) - 1
            
        is_indep = np.ones(max_mask, dtype=bool)
        
        for idx in triple_indices:
            tm = triple_masks[idx]
            is_indep[(all_masks_np & tm) == tm] = False
            
        # Find max popcount among independent sets
        scores = np.where(is_indep, popcounts, -1)
        max_alpha = int(np.max(scores))
        if max_alpha == 0:
             return 0, 0
             
        # Find the first mask with this score
        # Optimization: Precompute list of masks by popcount?
        # For now, linear scan. 65536 is small.
        for m in range(max_mask):
            if is_indep[m] and popcounts[m] == max_alpha:
                return max_alpha, m
                
        return max_alpha, 0

    best_alpha = n + 1
    best_indices = []
    
    start_time = time.time()
    time_limit = 115
    
    def greedy_packing(seed):
        random.seed(seed)
        shuffled_indices = list(range(num_triples))
        random.shuffle(shuffled_indices)
        
        cur_indices = []
        used_pairs = set()
        
        for idx in shuffled_indices:
            t = all_triples[idx]
            a, b, c = t
            p1 = (a, b) if a < b else (b, a)
            p2 = (a, c) if a < c else (c, a)
            p3 = (b, c) if b < c else (c, b)
            
            if p1 in used_pairs or p2 in used_pairs or p3 in used_pairs:
                continue
                
            cur_indices.append(idx)
            used_pairs.add(p1)
            used_pairs.add(p2)
            used_pairs.add(p3)
            
        return cur_indices, used_pairs

    # Phase 1: Find a good initial packing
    candidates = []
    for s in range(20):
        if time.time() - start_time > time_limit - 10:
            break
        inds, pairs = greedy_packing(s)
        if not inds:
            continue
        alpha, _ = calc_alpha_witness(inds)
        candidates.append((alpha, len(inds), inds, pairs))
        
    candidates.sort(key=lambda x: (x[0], -x[1]))
    
    if candidates:
        best_alpha, _, best_indices, best_pairs = candidates[0]
    else:
        best_indices, best_pairs = greedy_packing(0)
        best_alpha, _ = calc_alpha_witness(best_indices)

    # Phase 2: Local Search
    cur_indices = best_indices[:]
    cur_pairs = best_pairs
    cur_alpha = best_alpha
    
    # Precompute pairs for each triple index for faster access
    triple_pairs = []
    for t in all_triples:
        a, b, c = t
        p1 = (a, b) if a < b else (b, a)
        p2 = (a, c) if a < c else (c, a)
        p3 = (b, c) if b < c else (c, b)
        triple_pairs.append((p1, p2, p3))
        
    local_start = time.time()
    stall_count = 0
    
    while time.time() - local_start < time_limit - 2:
        alpha, witness_mask = calc_alpha_witness(cur_indices)
        
        if alpha < cur_alpha:
            cur_alpha = alpha
            stall_count = 0
            if cur_alpha < best_alpha:
                best_alpha = cur_alpha
                best_indices = cur_indices[:]
                if best_alpha <= 3:
                    break
        else:
            stall_count += 1
            if stall_count > 500:
                # Perturb: remove a random triple
                if cur_indices:
                    rem_idx = random.choice(cur_indices)
                    cur_indices.remove(rem_idx)
                    for p in triple_pairs[rem_idx]:
                        cur_pairs.discard(p)
                stall_count = 0
                continue

        if not cur_indices:
            break

        # Get witness points
        witness_points = [i for i in range(n) if (witness_mask >> i) & 1]
        
        # Generate all triples within witness
        w_triples = list(itertools.combinations(witness_points, 3))
        
        found_improvement = False
        random.shuffle(w_triples)
        
        for t in w_triples:
            t_idx = triple_to_idx.get(t)
            if t_idx is None:
                continue
            
            # Check if t is already in cur_indices
            if t_idx in cur_indices:
                continue
                
            p1, p2, p3 = triple_pairs[t_idx]
            
            # Check blockers
            blockers = set()
            if p1 in cur_pairs:
                # Find which triple has this pair
                # We need pair_to_triple map. 
                # Rebuilding it is O(N). N is small.
                # Or we can just check all triples in cur_indices?
                # Let's maintain a pair_to_triple dict.
                pass # Logic update needed below

        # This approach is getting complex to implement in a single SEARCH/REPLACE block 
        # without breaking the existing structure too much.
        # Let's stick to a simpler but effective improvement: 
        # Better Local Search with "Add if possible" and "Swap if improves".
        
        # 1. Try to add any valid triple (random)
        # 2. Try to swap a random triple with a random valid triple
        
        # Simple Local Search Step
        if random.random() < 0.5:
            # Try Add
            for _ in range(10):
                add_idx = random.randint(0, num_triples - 1)
                if add_idx in cur_indices:
                    continue
                p1, p2, p3 = triple_pairs[add_idx]
                if p1 in cur_pairs or p2 in cur_pairs or p3 in cur_pairs:
                    continue
                
                # Add
                cur_indices.append(add_idx)
                cur_pairs.add(p1)
                cur_pairs.add(p2)
                cur_pairs.add(p3)
                
                new_alpha, _ = calc_alpha_witness(cur_indices)
                if new_alpha <= cur_alpha:
                    cur_alpha = new_alpha
                    found_improvement = True
                    break
                else:
                    # Revert
                    cur_indices.pop()
                    cur_pairs.discard(p1)
                    cur_pairs.discard(p2)
                    cur_pairs.discard(p3)
        else:
            # Try Swap
            if not cur_indices:
                continue
            rem_idx_pos = random.randint(0, len(cur_indices) - 1)
            rem_idx = cur_indices[rem_idx_pos]
            
            # Remove
            rp1, rp2, rp3 = triple_pairs[rem_idx]
            cur_indices.pop(rem_idx_pos)
            cur_pairs.discard(rp1)
            cur_pairs.discard(rp2)
            cur_pairs.discard(rp3)
            
            added = False
            for _ in range(10):
                add_idx = random.randint(0, num_triples - 1)
                if add_idx in cur_indices:
                    continue
                ap1, ap2, ap3 = triple_pairs[add_idx]
                if ap1 in cur_pairs or ap2 in cur_pairs or ap3 in cur_pairs:
                    continue
                
                # Add
                cur_indices.append(add_idx)
                cur_pairs.add(ap1)
                cur_pairs.add(ap2)
                cur_pairs.add(ap3)
                added = True
                break
                
            if not added:
                # Revert removal
                cur_indices.insert(rem_idx_pos, rem_idx)
                cur_pairs.add(rp1)
                cur_pairs.add(rp2)
                cur_pairs.add(rp3)
                continue
                
            new_alpha, _ = calc_alpha_witness(cur_indices)
            if new_alpha < cur_alpha:
                cur_alpha = new_alpha
                found_improvement = True
            elif new_alpha == cur_alpha and random.random() < 0.1:
                # Accept neutral move to explore
                found_improvement = True
            else:
                # Revert
                cur_indices.pop()
                cur_pairs.discard(ap1)
                cur_pairs.discard(ap2)
                cur_pairs.discard(ap3)
                cur_indices.insert(rem_idx_pos, rem_idx)
                cur_pairs.add(rp1)
                cur_pairs.add(rp2)
                cur_pairs.add(rp3)

    return [list(all_triples[i]) for i in best_indices]
