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
        
    # Precompute masks for all subsets of size k
    subset_masks = {}
    for k in range(1, n + 1):
        masks = []
        for subset in itertools.combinations(range(n), k):
            mask = 0
            for p in subset:
                mask |= (1 << p)
            masks.append(mask)
        subset_masks[k] = masks
        
    # Precompute conflict sets: triples that share a pair
    conflicting_triples = {t: set() for t in all_triples}
    for i in range(len(all_triples)):
        for j in range(i + 1, len(all_triples)):
            t1 = all_triples[i]
            t2 = all_triples[j]
            if len(set(t1) & set(t2)) >= 2:
                conflicting_triples[t1].add(t2)
                conflicting_triples[t2].add(t1)

    def calculate_alpha_and_set(triples, max_k=n):
        """
        Calculate the exact independence number and return (alpha, set).
        Only checks k from max_k down to 1.
        """
        if not triples:
            return max_k, set(range(max_k))
            
        triple_m = [triple_masks[t] for t in triples]
        
        for k in range(min(max_k, n), 0, -1):
            masks = subset_masks[k]
            for mask in masks:
                is_independent = True
                for tm in triple_m:
                    if (tm & mask) == tm:
                        is_independent = False
                        break
                if is_independent:
                    s = set()
                    for p in range(n):
                        if mask & (1 << p):
                            s.add(p)
                    return k, s
        return 1, {0}

    def is_compatible(t_new, system_set, exclude_t_old):
        """
        Check if t_new is compatible with all triples in system_set except exclude_t_old.
        """
        conflicts = conflicting_triples[t_new]
        for c in conflicts:
            if c in system_set and c != exclude_t_old:
                return False
        return True

    best_alpha = 17
    best_triples = []
    rng = random.Random(42)
    
    def generate_maximal_packing(order):
        used_pairs = set()
        triples = []
        for t in order:
            ps = triple_pairs[t]
            if not (ps & used_pairs):
                triples.append(t)
                used_pairs.update(ps)
        return triples

    def local_search(triples, time_limit):
        nonlocal best_alpha, best_triples
        
        system_set = set(triples)
        current_alpha, current_set = calculate_alpha_and_set(triples, max_k=n)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_triples = triples[:]
            
        while time.time() < time_limit:
            # Find candidates: triples subset of current_set
            candidates = [t for t in all_triples if t[0] in current_set and t[1] in current_set and t[2] in current_set]
            
            if not candidates:
                break
                
            rng.shuffle(candidates)
            
            improvement_made = False
            for t_new in candidates:
                # Pick a random t_old to remove
                idx = rng.randrange(len(triples))
                t_old = triples[idx]
                
                if t_new == t_old:
                    continue
                    
                if is_compatible(t_new, system_set, t_old):
                    # Perform swap
                    triples[idx] = t_new
                    system_set.remove(t_old)
                    system_set.add(t_new)
                    
                    # Recalculate alpha, knowing it cannot be larger than current_alpha
                    new_alpha, new_set = calculate_alpha_and_set(triples, max_k=current_alpha)
                    
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_triples = triples[:]
                        
                    if new_alpha < current_alpha:
                        current_alpha = new_alpha
                        current_set = new_set
                        improvement_made = True
                        break
                    else:
                        # Revert
                        triples[idx] = t_old
                        system_set.remove(t_new)
                        system_set.add(t_old)
                        
            if not improvement_made:
                break
                
            if best_alpha <= 5:
                return

    # Main Loop
    for i in range(100):
        if time.time() > deadline:
            break
            
        order = all_triples[:]
        rng.shuffle(order)
        triples = generate_maximal_packing(order)
        
        current_alpha, _ = calculate_alpha_and_set(triples, max_k=n)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_triples = triples[:]
            
        search_deadline = min(time.time() + 2.0, deadline - 5.0)
        if search_deadline > time.time():
            local_search(triples, search_deadline)
            
        if best_alpha <= 5:
            break
            
    return best_triples
