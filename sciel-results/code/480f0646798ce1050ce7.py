import itertools
import random
import time

def construct():
    deadline = time.time() + 145
    n = 16
    points = list(range(n))
    all_triples = list(itertools.combinations(points, 3))
    
    triple_pairs = {}
    for t in all_triples:
        a, b, c = t
        triple_pairs[t] = frozenset({(a, b), (a, c), (b, c)})
        
    best_alpha = 17
    best_system = []
    
    def calc_alpha(triples):
        if not triples:
            return n
        t_masks = []
        for t in triples:
            a, b, c = t
            t_masks.append((1 << a) | (1 << b) | (1 << c))
            
        for k in range(n, 0, -1):
            for subset in itertools.combinations(points, k):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                is_indep = True
                for tm in t_masks:
                    if (mask & tm) == tm:
                        is_indep = False
                        break
                if is_indep:
                    return k
        return 0

    # Phase 1: Random Greedy
    while time.time() < deadline - 10: # Leave 10s for local search
        random.shuffle(all_triples)
        current_triples = []
        used_pairs = set()
        
        for t in all_triples:
            pairs = triple_pairs[t]
            if not any(p in used_pairs for p in pairs):
                current_triples.append(t)
                used_pairs.update(pairs)
                
        alpha = calc_alpha(current_triples)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = current_triples[:]
            
        if best_alpha <= 6:
            break

    # Phase 2: Local Search on best_system
    if best_system and time.time() < deadline - 2:
        current = best_system[:]
        current_alpha = best_alpha
        used_pairs = set()
        for t in current:
            used_pairs.update(triple_pairs[t])
            
        # Precompute valid triples for quick check?
        # Just check linearity on the fly.
        
        while time.time() < deadline - 2:
            # Pick a random triple to remove
            idx = random.randint(0, len(current) - 1)
            t_out = current[idx]
            
            # Remove t_out
            current.pop(idx)
            for p in triple_pairs[t_out]:
                used_pairs.discard(p)
                
            # Find a triple to add
            # Try random triples from all_triples
            # To make it faster, we can pre-filter triples that don't conflict with used_pairs
            # But used_pairs changes.
            
            # Simple approach: try random triples until one fits
            added = False
            for _ in range(100): # Try 100 random candidates
                t_in = random.choice(all_triples)
                if t_in in current:
                    continue
                pairs_in = triple_pairs[t_in]
                if not any(p in used_pairs for p in pairs_in):
                    current.append(t_in)
                    used_pairs.update(pairs_in)
                    added = True
                    break
            
            if not added:
                # Revert
                current.insert(idx, t_out)
                for p in triple_pairs[t_out]:
                    used_pairs.add(p)
                continue
            
            # Evaluate new state
            new_alpha = calc_alpha(current)
            
            if new_alpha < current_alpha:
                current_alpha = new_alpha
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_system = current[:]
            else:
                # Revert
                t_in = current.pop()
                for p in triple_pairs[t_in]:
                    used_pairs.discard(p)
                current.insert(idx, t_out)
                for p in triple_pairs[t_out]:
                    used_pairs.add(p)

    return best_system
