import itertools
import random
import time

def construct():
    n = 16
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

    def calculate_alpha(triples):
        """
        Calculate the exact independence number.
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

    best_alpha = 17
    best_triples = []
    rng = random.Random(42)
    
    def generate_random_packing():
        used_pairs = set()
        triples = []
        candidates = all_triples[:]
        rng.shuffle(candidates)
        
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
                
            idx = rng.randrange(len(triples))
            t_old = triples[idx]
            ps_old = triple_pairs[t_old]
            
            # Remove t_old
            used_pairs.difference_update(ps_old)
            
            # Find valid replacements
            # Optimization: Pre-compute a list of valid candidates? 
            # Or just iterate all_triples and check.
            # all_triples has 560 items. Checking 560 items is fast.
            
            valid_candidates = []
            for t_new in all_triples:
                if t_new == t_old:
                    continue
                ps_new = triple_pairs[t_new]
                if not (ps_new & used_pairs):
                    valid_candidates.append(t_new)
            
            if not valid_candidates:
                # Revert
                used_pairs.update(ps_old)
                triples[idx] = t_old
                continue
            
            # Pick a random candidate
            t_new = rng.choice(valid_candidates)
            ps_new = triple_pairs[t_new]
            
            # Apply move
            triples[idx] = t_new
            used_pairs.update(ps_new)
            
            # Calculate new alpha
            new_alpha = calculate_alpha(triples)
            
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best_triples = triples[:]
                
            # Acceptance criteria
            if new_alpha <= current_alpha:
                current_alpha = new_alpha
            else:
                # Revert
                used_pairs.difference_update(ps_new)
                triples[idx] = t_old
                used_pairs.update(ps_old)
                
            if best_alpha <= 4:
                return

    # Main Loop
    while time.time() < deadline:
        triples, used_pairs = generate_random_packing()
        current_alpha = calculate_alpha(triples)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_triples = triples[:]
        
        # Run local search for a short duration
        # Use a small time slice to allow for many restarts
        search_deadline = min(time.time() + 0.5, deadline - 1.0)
        if search_deadline > time.time():
            local_search(triples, used_pairs, current_alpha, search_deadline)
        
        if best_alpha <= 4:
            break

    return best_triples
