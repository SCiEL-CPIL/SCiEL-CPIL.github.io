import time
import random
import itertools
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    
    def is_linear(triples):
        """Check if any two triples share at most one point."""
        for i in range(len(triples)):
            for j in range(i+1, len(triples)):
                common = set(triples[i]) & set(triples[j])
                if len(common) > 1:
                    return False
        return True
    
    def compute_alpha(triples):
        """Find the independence number (largest set with no triple)."""
        triple_set = set(triples)
        # Check from size 16 down
        for size in range(16, 0, -1):
            for subset in itertools.combinations(points, size):
                s = set(subset)
                # Check if any triple is fully contained
                has_triple = False
                for t in triple_set:
                    if set(t).issubset(s):
                        has_triple = True
                        break
                if not has_triple:
                    return size
        return 0
    
    def compute_alpha_fast(triples):
        """Faster alpha computation using bitmask."""
        n = len(triples)
        # For each subset, check if it contains any triple
        # Use bitmasks for triples
        triple_masks = []
        for t in triples:
            m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            triple_masks.append(m)
        
        for size in range(16, 0, -1):
            for subset in itertools.combinations(points, size):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                # Check if any triple mask is subset of this mask
                found = False
                for tm in triple_masks:
                    if (tm & mask) == tm:
                        found = True
                        break
                if not found:
                    return size
        return 0
    
    # Greedy construction of linear triple system
    def greedy_construct(rng):
        triples = []
        used_pairs = set()
        
        # Shuffle all triples
        shuffled = all_triples[:]
        rng.shuffle(shuffled)
        
        for t in shuffled:
            # Check linearity: any pair in t should not appear in another triple
            a, b, c = t
            pairs = [(a,b), (a,c), (b,c)]
            ok = True
            for p in pairs:
                if p in used_pairs:
                    ok = False
                    break
            if ok:
                triples.append(t)
                for p in pairs:
                    used_pairs.add(p)
        
        return triples
    
    # Local search to minimize alpha
    def local_search(triples, rng, max_iters=5000):
        triples = list(triples)
        current_alpha = compute_alpha_fast(triples)
        best_triples = triples[:]
        best_alpha = current_alpha
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if current_alpha <= 5:  # Can't do better than ~5
                break
            
            # Find a maximum independent set
            # Actually, let's find any independent set of size current_alpha
            triple_set = set(map(tuple, triples))
            triple_masks = []
            for t in triples:
                m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
                triple_masks.append(m)
            
            # Find an independent set of size current_alpha
            indep_set = None
            for subset in itertools.combinations(points, current_alpha):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                found = False
                for tm in triple_masks:
                    if (tm & mask) == tm:
                        found = True
                        break
                if not found:
                    indep_set = subset
                    break
            
            if indep_set is None:
                # Recompute alpha
                current_alpha = compute_alpha_fast(triples)
                continue
            
            # Try to add a triple that hits this independent set in 2 or 3 points
            # The triple must be linear with existing triples
            a, b, c = indep_set[0], indep_set[1], indep_set[2]
            candidate = (a, b, c)
            
            # Check if candidate is linear with existing triples
            can_add = True
            for t in triples:
                common = set(candidate) & set(t)
                if len(common) > 1:
                    can_add = False
                    break
            
            if can_add:
                triples.append(candidate)
                new_alpha = compute_alpha_fast(triples)
                if new_alpha < current_alpha:
                    current_alpha = new_alpha
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_triples = triples[:]
                else:
                    triples.pop()
            else:
                # Try other combinations from the independent set
                for combo in itertools.combinations(indep_set, 3):
                    candidate = combo
                    can_add = True
                    for t in triples:
                        common = set(candidate) & set(t)
                        if len(common) > 1:
                            can_add = False
                            break
                    if can_add:
                        triples.append(candidate)
                        new_alpha = compute_alpha_fast(triples)
                        if new_alpha < current_alpha:
                            current_alpha = new_alpha
                            if current_alpha < best_alpha:
                                best_alpha = current_alpha
                                best_triples = triples[:]
                        else:
                            triples.pop()
                        break
        
        return best_triples, best_alpha
    
    # Main search
    rng = random.Random(42)
    best_triples = None
    best_alpha = 16
    
    # Try multiple greedy constructions
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        triples = greedy_construct(rng)
        alpha = compute_alpha_fast(triples)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples[:]
        
        # Local search
        if time.time() < deadline - 10:
            ls_triples, ls_alpha = local_search(triples, rng, max_iters=2000)
            if ls_alpha < best_alpha:
                best_alpha = ls_alpha
                best_triples = ls_triples
    
    if best_triples is None:
        best_triples = greedy_construct(rng)
    
    # Final verification
    if not is_linear(best_triples):
        # Fallback
        best_triples = [[3*i, 3*i+1, 3*i+2] for i in range(5)]
    
    return [list(t) for t in best_triples]
