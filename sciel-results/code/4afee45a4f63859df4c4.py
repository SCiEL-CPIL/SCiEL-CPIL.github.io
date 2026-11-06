import itertools
import random
import time

def construct():
    n = 16
    # Safety margin: 120s limit, use 115s
    deadline = time.time() + 115.0
    
    # Precompute all triples
    all_triples = list(itertools.combinations(range(n), 3))
    
    # Precompute pairs for each triple
    triple_pairs = {}
    for t in all_triples:
        a, b, c = t
        triple_pairs[t] = frozenset([(a, b), (a, c), (b, c)])
        
    # Precompute bitmasks for triples
    triple_masks = {}
    for t in all_triples:
        a, b, c = t
        triple_masks[t] = (1 << a) | (1 << b) | (1 << c)
    
    # Precompute masks for all subsets
    subset_masks = {}
    for k in range(1, n + 1):
        masks = []
        for subset in itertools.combinations(range(n), k):
            mask = 0
            for p in subset:
                mask |= (1 << p)
            masks.append(mask)
        subset_masks[k] = masks

    def calculate_alpha_fast(triples):
        """
        Calculate the exact independence number using bitmask operations.
        Returns the size of the largest independent set.
        """
        if not triples:
            return n
            
        triple_m = [triple_masks[t] for t in triples]
        
        for k in range(16, 0, -1):
            masks = subset_masks[k]
            for mask in masks:
                is_independent = True
                for tm in triple_m:
                    if (tm & mask) == tm:
                        is_independent = False
                        break
                if is_independent:
                    return k
        return 1

    def find_largest_independent_set(triples):
        """
        Find the actual largest independent set (as a mask).
        """
        if not triples:
            return (1 << n) - 1
            
        triple_m = [triple_masks[t] for t in triples]
        
        for k in range(16, 0, -1):
            masks = subset_masks[k]
            for mask in masks:
                is_independent = True
                for tm in triple_m:
                    if (tm & mask) == tm:
                        is_independent = False
                        break
                if is_independent:
                    return mask
        return 0

    best_alpha = 17
    best_triples = []
    rng = random.Random(42)
    
    def generate_random_packing(seed_offset=0):
        used_pairs = set()
        triples = []
        candidates = all_triples[:]
        rng_local = random.Random(42 + seed_offset)
        rng_local.shuffle(candidates)
        
        for t in candidates:
            ps = triple_pairs[t]
            if not (ps & used_pairs):
                triples.append(t)
                used_pairs.update(ps)
        
        return triples, used_pairs

    def local_search(triples, used_pairs, current_alpha, time_limit):
        nonlocal best_alpha, best_triples
        
        while time.time() < time_limit:
            if not triples:
                break
                
            # Find the largest independent set to target
            indep_mask = find_largest_independent_set(triples)
            if indep_mask == 0:
                break
                
            # Count how many points are in the independent set
            indep_count = bin(indep_mask).count('1')
            
            # Strategy: Remove a triple that shares points with the independent set
            # to "break up" the independent set
            candidates_to_remove = []
            for idx, t in enumerate(triples):
                tm = triple_masks[t]
                shared = tm & indep_mask
                if shared:
                    # Score: prefer triples that share more points with independent set
                    candidates_to_remove.append((bin(shared).count('1'), idx))
            
            if not candidates_to_remove:
                # No triple intersects the independent set, try random removal
                idx = rng.randrange(len(triples))
            else:
                # Pick the triple that shares the most points with the independent set
                candidates_to_remove.sort(reverse=True)
                idx = candidates_to_remove[0][1]
            
            t_old = triples[idx]
            ps_old = triple_pairs[t_old]
            
            # Remove t_old
            used_pairs.difference_update(ps_old)
            
            # Find valid replacement that helps cover the independent set
            candidates = all_triples[:]
            rng.shuffle(candidates)
            
            found = False
            for t_new in candidates:
                if t_new == t_old:
                    continue
                ps_new = triple_pairs[t_new]
                if not (ps_new & used_pairs):
                    # Check if this replacement helps reduce the independent set
                    tm_new = triple_masks[t_new]
                    shared_with_indep = tm_new & indep_mask
                    
                    triples[idx] = t_new
                    used_pairs.update(ps_new)
                    
                    new_alpha = calculate_alpha_fast(triples)
                    
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_triples = triples[:]
                    
                    # Accept move if it doesn't increase alpha
                    if new_alpha <= current_alpha:
                        current_alpha = new_alpha
                    else:
                        # Revert
                        used_pairs.difference_update(ps_new)
                        triples[idx] = t_old
                        used_pairs.update(ps_old)
                    
                    found = True
                    break
            
            if not found:
                # Revert
                used_pairs.update(ps_old)
                triples[idx] = t_old
                
            if best_alpha <= 5:
                return

    # Main Loop
    while time.time() < deadline:
        seed_offset = int(time.time() * 1000) % 1000
        triples, used_pairs = generate_random_packing(seed_offset)
        current_alpha = calculate_alpha_fast(triples)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_triples = triples[:]
        
        # Run local search for a short duration
        search_deadline = min(time.time() + 1.0, deadline - 2.0)
        if search_deadline > time.time():
            local_search(triples, used_pairs, current_alpha, search_deadline)
        
        if best_alpha <= 5:
            break

    return best_triples
