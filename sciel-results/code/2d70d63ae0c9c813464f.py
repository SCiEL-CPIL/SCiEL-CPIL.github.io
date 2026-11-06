"""Linear triple systems on 16 points - minimize independence number."""
import itertools
import random
import time
import numpy as np

def construct():
    deadline = time.time() + 145
    n = 16
    rng = random.Random(42)
    
    all_triples = list(itertools.combinations(range(n), 3))
    rng.shuffle(all_triples)
    
    def get_pairs(triple):
        a, b, c = triple
        return frozenset({(a, b), (a, c), (b, c)})
    
    def is_linear(triples):
        seen_pairs = set()
        for t in triples:
            ps = get_pairs(t)
            if seen_pairs & ps:
                return False
            seen_pairs |= ps
        return True
    
    def independence_number(triples):
        """Find largest subset of points containing no triple."""
        triple_set = set(tuple(sorted(t)) for t in triples)
        # Check subsets from largest to smallest
        for size in range(n, 0, -1):
            for subset in itertools.combinations(range(n), size):
                s = set(subset)
                has_triple = False
                for t in triple_set:
                    if set(t) <= s:
                        has_triple = True
                        break
                if not has_triple:
                    return size
        return 0
    
    def independence_number_fast(triples):
        """Faster independence number using bitmasks."""
        # Represent each triple as a bitmask
        triple_masks = []
        for t in triples:
            mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            triple_masks.append(mask)
        
        for size in range(n, 0, -1):
            for subset in itertools.combinations(range(n), size):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                # Check if any triple is fully contained
                found = False
                for tm in triple_masks:
                    if (tm & mask) == tm:
                        found = True
                        break
                if not found:
                    return size
        return 0
    
    # Greedy construction
    def greedy_construct():
        triples = []
        used_pairs = set()
        for t in all_triples:
            ps = get_pairs(t)
            if not (used_pairs & ps):
                triples.append(t)
                used_pairs |= ps
        return triples
    
    # Local search: swap one triple for another
    def local_search(triples, time_limit):
        best_triples = list(triples)
        best_alpha = independence_number_fast(best_triples)
        start = time.time()
        
        while time.time() < time_limit:
            improved = False
            current_alpha = best_alpha
            
            # Try removing one triple and adding another
            for i in range(len(best_triples)):
                if time.time() >= time_limit:
                    break
                candidate = best_triples[:i] + best_triples[i+1:]
                used_pairs = set()
                for t in candidate:
                    used_pairs |= get_pairs(t)
                
                # Try adding a new triple
                for t in all_triples:
                    if t in candidate:
                        continue
                    ps = get_pairs(t)
                    if used_pairs & ps:
                        continue
                    new_candidate = candidate + [t]
                    new_alpha = independence_number_fast(new_candidate)
                    if new_alpha < current_alpha:
                        best_triples = new_candidate
                        best_alpha = new_alpha
                        current_alpha = new_alpha
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                # Try adding a triple if we can
                used_pairs = set()
                for t in best_triples:
                    used_pairs |= get_pairs(t)
                
                added = False
                for t in all_triples:
                    if t in best_triples:
                        continue
                    ps = get_pairs(t)
                    if not (used_pairs & ps):
                        new_candidate = best_triples + [t]
                        new_alpha = independence_number_fast(new_candidate)
                        if new_alpha <= best_alpha:
                            best_triples = new_candidate
                            best_alpha = new_alpha
                            added = True
                            break
                
                if not added:
                    # Try 2-opt: remove two, add two
                    if len(best_triples) >= 2:
                        found = False
                        for i in range(len(best_triples)):
                            if time.time() >= time_limit:
                                break
                            for j in range(i+1, len(best_triples)):
                                if time.time() >= time_limit:
                                    break
                                candidate = [t for k, t in enumerate(best_triples) if k != i and k != j]
                                used_pairs = set()
                                for t in candidate:
                                    used_pairs |= get_pairs(t)
                                
                                for t1 in all_triples:
                                    if t1 in candidate:
                                        continue
                                    ps1 = get_pairs(t1)
                                    if used_pairs & ps1:
                                        continue
                                    used_pairs_new = used_pairs | ps1
                                    for t2 in all_triples:
                                        if t2 in candidate or t2 == t1:
                                            continue
                                        ps2 = get_pairs(t2)
                                        if (used_pairs_new & ps2) or (ps1 & ps2):
                                            continue
                                        new_candidate = candidate + [t1, t2]
                                        new_alpha = independence_number_fast(new_candidate)
                                        if new_alpha < best_alpha:
                                            best_triples = new_candidate
                                            best_alpha = new_alpha
                                            found = True
                                            break
                                    if found:
                                        break
                                if found:
                                    break
                            if found:
                                break
                        
                        if not found:
                            break
                    else:
                        break
        
        return best_triples, best_alpha
    
    # Initial greedy construction
    triples = greedy_construct()
    
    # Run local search
    triples, alpha = local_search(triples, deadline - 5)
    
    return [list(t) for t in triples]
