import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    n = 16
    all_points = list(range(n))
    all_triples = list(itertools.combinations(range(n), 3))
    triple_to_mask = {t: (1 << t[0]) | (1 << t[1]) | (1 << t[2]) for t in all_triples}
    
    # Precompute masks for all subsets? No, too much memory? 2^16 is fine.
    # But we need to check independence.
    
    def calc_alpha(triple_set):
        # triple_set is a set of tuples
        if not triple_set:
            return n
        
        masks = [triple_to_mask[t] for t in triple_set]
        
        # Check sizes from n down to 1
        # We can optimize by checking if any subset of size k is independent.
        # A subset S is independent if no mask is a subset of S.
        
        # Generate all masks of size k
        # It is faster to iterate all 2^n masks and check?
        # 65536 masks.
        # For each mask, check against all triple masks.
        # If M=30, 65536 * 30 = 2M ops.
        
        # Let's try iterating by size to stop early if we find a large one.
        # But we want the MAX independent set.
        # So we check size 16, 15, ...
        
        # Actually, checking size 16 is just 1 mask.
        # Size 15 is 16 masks.
        # Size 14 is 120 masks.
        # ...
        # Size 8 is 12870.
        
        # If alpha is small (e.g. 6), we check sizes 16..7.
        # Sum of binomial coeffs for 16..7 is half of 2^16 approx.
        
        for size in range(n, 0, -1):
            for subset in itertools.combinations(range(n), size):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                
                is_indep = True
                for tm in masks:
                    if (mask & tm) == tm:
                        is_indep = False
                        break
                if is_indep:
                    return size
        return 0

    best_alpha = n
    best_triples = []
    
    # Greedy Randomized Construction
    for _ in range(5000):
        if time.time() > deadline:
            break
            
        random.shuffle(all_triples)
        current_triples = []
        used_pairs = set()
        
        for t in all_triples:
            # Check linearity: any two triples share at most one point.
            # This is equivalent to no pair of points appears in more than one triple.
            a, b, c = t
            pairs = frozenset({(a,b), (a,c), (b,c)})
            # Note: pairs are unordered? (a,b) and (b,a) are same.
            # Let's normalize pairs to (min, max)
            p1 = (min(a,b), max(a,b))
            p2 = (min(a,c), max(a,c))
            p3 = (min(b,c), max(b,c))
            p_set = {p1, p2, p3}
            
            if used_pairs.isdisjoint(p_set):
                current_triples.append(t)
                used_pairs.update(p_set)
        
        if not current_triples:
            continue
            
        alpha = calc_alpha(set(current_triples))
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = current_triples.copy()
            
            if best_alpha <= 5: # Heuristic lower bound?
                # Try to break early if we hit a very good value
                # But let's keep searching for a bit more to be sure
                pass

    # Local Search on best found
    if best_triples:
        current_set = set(best_triples)
        current_alpha = best_alpha
        
        # Try to improve
        for _ in range(1000):
            if time.time() > deadline:
                break
                
            # Pick a random triple to remove
            if not current_set:
                break
            t_out = random.choice(list(current_set))
            new_set = current_set - {t_out}
            
            # Try to add a random triple not in set
            # We need a pool of candidates
            # Just pick random from all_triples
            for _ in range(10): # Try a few candidates
                t_in = random.choice(all_triples)
                if t_in in new_set:
                    continue
                
                a, b, c = t_in
                p1 = (min(a,b), max(a,b))
                p2 = (min(a,c), max(a,c))
                p3 = (min(b,c), max(b,c))
                p_set = {p1, p2, p3}
                
                # Check if t_in is compatible with new_set
                # We need to check against all triples in new_set
                # This is O(N) per check
                compatible = True
                for t_existing in new_set:
                    ea, eb, ec = t_existing
                    ep1 = (min(ea,eb), max(ea,eb))
                    ep2 = (min(ea,ec), max(ea,ec))
                    ep3 = (min(eb,ec), max(eb,ec))
                    if p_set & {ep1, ep2, ep3}:
                        compatible = False
                        break
                
                if compatible:
                    candidate_set = new_set | {t_in}
                    # Only calculate alpha if size changed or we want to be precise
                    # If we just swap, size is same.
                    alpha_new = calc_alpha(candidate_set)
                    if alpha_new < current_alpha:
                        current_set = candidate_set
                        current_alpha = alpha_new
                        best_alpha = current_alpha
                        best_triples = list(current_set)
                    break # Break inner loop, try next outer iteration
                    
    return best_triples
