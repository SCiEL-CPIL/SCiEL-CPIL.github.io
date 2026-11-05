import time
import random
from itertools import combinations

DEADLINE = 140.0

def _calc_alpha(triples, n=16):
    """
    Calculates the independence number of the hypergraph defined by triples.
    Returns the size of the largest subset of points containing no triple.
    """
    if not triples:
        return n
        
    triple_masks = []
    for t in triples:
        m = 0
        for x in t:
            m |= (1 << x)
        triple_masks.append(m)
        
    # Check subset sizes from n down to 1
    # If we find an independent set of size k, that is the alpha.
    for k in range(n, 0, -1):
        for combo in combinations(range(n), k):
            mask = 0
            for x in combo:
                mask |= (1 << x)
            
            # Check if any triple is fully contained in this mask
            is_independent = True
            for tm in triple_masks:
                if (mask & tm) == tm:
                    is_independent = False
                    break
            
            if is_independent:
                return k
    return 0

def _is_linear(triples):
    """
    Checks if the system is linear: any two triples share at most one point.
    """
    pairs = set()
    for t in triples:
        ts = sorted(t)
        for a, b in combinations(ts, 2):
            if (a, b) in pairs:
                return False
            pairs.add((a, b))
    return True

def _greedy_packing(n=16, seed=0):
    """
    Generates a linear triple system using a greedy approach with randomization.
    """
    random.seed(seed)
    all_triples = list(combinations(range(n), 3))
    random.shuffle(all_triples)
    
    selected = []
    used_pairs = set()
    
    for t in all_triples:
        ts = sorted(t)
        conflict = False
        for a, b in combinations(ts, 2):
            if (a, b) in used_pairs:
                conflict = True
                break
        if not conflict:
            selected.append(list(t))
            for a, b in combinations(ts, 2):
                used_pairs.add((a, b))
                
    return selected

def construct():
    start_time = time.time()
    n = 16
    
    def time_ok():
        return time.time() - start_time < DEADLINE

    best_triples = None
    best_alpha = n + 1
    
    # Phase 1: Generate multiple diverse initial packings
    # Run greedy packing with different seeds
    for seed in range(200):
        if not time_ok():
            break
        triples = _greedy_packing(n, seed)
        if _is_linear(triples):
            alpha = _calc_alpha(triples, n)
            if alpha < best_alpha:
                best_alpha = alpha
                best_triples = triples
                # If we find a very good alpha (e.g., <= 7), we can stop early?
                # Theoretical lower bound for n=16 is likely around 7 or 8.
                # Let's keep searching to see if we can get 7.

    if best_triples is None:
        best_triples = _greedy_packing(n, 0)
        best_alpha = _calc_alpha(best_triples, n)
        
    current = [list(t) for t in best_triples]
    cur_alpha = best_alpha
    
    all_triples = list(combinations(range(n), 3))
    random.seed(123)
    
    # Phase 2: Local Search
    # We try to improve the alpha.
    # Moves: Add, Remove, Swap.
    
    no_improve_count = 0
    
    while time_ok():
        move_type = random.random()
        new_current = [list(t) for t in current]
        changed = False
        
        if move_type < 0.4:
            # Add a random triple
            t = list(random.choice(all_triples))
            can_add = True
            for t2 in current:
                if len(set(t) & set(t2)) > 1:
                    can_add = False
                    break
            if can_add:
                new_current.append(t)
                changed = True
                
        elif move_type < 0.7:
            # Remove a random triple
            if len(current) > 0:
                idx = random.randint(0, len(current) - 1)
                new_current.pop(idx)
                changed = True
        else:
            # Swap: remove one, add one
            if len(current) > 0:
                idx = random.randint(0, len(current) - 1)
                removed = new_current.pop(idx)
                t = list(random.choice(all_triples))
                can_add = True
                for t2 in new_current:
                    if len(set(t) & set(t2)) > 1:
                        can_add = False
                        break
                if can_add:
                    new_current.append(t)
                    changed = True
                else:
                    new_current.append(removed)
                    changed = False # No change if we revert
        
        if not changed:
            continue
            
        if _is_linear(new_current):
            new_alpha = _calc_alpha(new_current, n)
            # Accept if alpha improves, or if alpha is same and we have more triples
            if new_alpha < cur_alpha or (new_alpha == cur_alpha and len(new_current) > len(current)):
                current = new_current
                cur_alpha = new_alpha
                if cur_alpha < best_alpha:
                    best_alpha = cur_alpha
                    best_triples = current
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
            
        # If stuck for a long time, restart from a new random greedy
        if no_improve_count > 500:
            no_improve_count = 0
            new_seed = random.randint(0, 10000)
            triples = _greedy_packing(n, new_seed)
            if _is_linear(triples):
                new_alpha = _calc_alpha(triples, n)
                if new_alpha < cur_alpha:
                    current = triples
                    cur_alpha = new_alpha
                    if cur_alpha < best_alpha:
                        best_alpha = cur_alpha
                        best_triples = current

    # Final safety check
    if not _is_linear(best_triples):
        # Fallback to a simple valid system
        return [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
        
    return best_triples

if __name__ == "__main__":
    res = construct()
    print("triples:", len(res), "alpha:", _calc_alpha(res))
