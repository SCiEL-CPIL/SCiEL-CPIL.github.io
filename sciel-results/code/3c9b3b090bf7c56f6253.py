import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 130.0

    N = 16

    def time_check():
        return time.time() - start_time > DEADLINE

    def compute_alpha_fast(triples):
        """Compute independence number using precomputed 'contains triple' table."""
        if not triples:
            return N
        
        # Precompute triple masks
        triple_masks = []
        for t in triples:
            m = 0
            for p in t:
                m |= (1 << p)
            triple_masks.append(m)
        
        n_tm = len(triple_masks)
        
        # For each mask (0 to 2^16-1), determine if it contains any triple
        # A mask is independent if it does NOT contain any triple
        # contains[i] = True if mask i contains at least one triple
        
        # Efficient approach: for each triple mask, mark all supersets
        # But 2^16 * n_tm might be too slow. Instead, iterate masks and check.
        
        # Better: precompute for each point, which triples it's in
        # Then for a candidate independent set S, check that no triple has all 3 points in S
        
        # Fastest: iterate all masks, use bit tricks
        # For each mask, check if any triple_mask is subset of mask
        # (mask & triple_mask) == triple_mask means triple is contained
        
        best_indep = 0
        full_mask = (1 << N) - 1
        
        # Iterate masks by decreasing popcount to find max early
        # But we need to be efficient. Let's just check all masks.
        # 65536 masks * ~40 triples = ~2.6M ops, should be fast enough
        
        for mask in range(full_mask, -1, -1):
            popc = bin(mask).count('1')
            if popc <= best_indep:
                continue
            # Check if mask is independent (contains no triple)
            independent = True
            for tm in triple_masks:
                if (mask & tm) == tm:
                    independent = False
                    break
            if independent:
                best_indep = popc
                # Early exit if we can't do better
                if best_indep >= 8:
                    break
        
        return best_indep

    def backtracking_construct(seed, target_alpha=None):
        """Backtracking construction aiming for dense linear triple system."""
        rng = random.Random(seed)
        
        # All possible triples on 16 points
        all_triples = list(combinations(range(N), 3))
        
        # Shuffle with some bias toward balanced point usage
        # Compute degree of each point
        point_deg = [0] * N
        for t in all_triples:
            for p in t:
                point_deg[p] += 1
        
        # Score triples: prefer those using low-degree points
        def triple_score(t):
            return sum(point_deg[p] for p in t)
        
        all_triples.sort(key=triple_score)
        rng.shuffle(all_triples)
        
        used_pairs = set()
        triples = []
        
        # Try to add triples, preferring those that cover underused points
        for t in all_triples:
            if time_check():
                break
            a, b, c = t
            pairs = frozenset((a, b)), frozenset((a, c)), frozenset((b, c))
            
            conflict = False
            for p in pairs:
                if p in used_pairs:
                    conflict = True
                    break
            
            if not conflict:
                triples.append(list(t))
                for p in pairs:
                    used_pairs.add(p)
        
        return triples

    def local_search_add(triples, max_iters=100):
        """Local search: try to add triples that reduce independence number."""
        if not triples:
            return triples
        
        current = [list(t) for t in triples]
        current_alpha = compute_alpha_fast(current)
        
        if current_alpha <= 5:
            return current
        
        # Build pair set for fast lookup
        used_pairs = set()
        for t in current:
            a, b, c = t
            used_pairs.add(frozenset((a, b)))
            used_pairs.add(frozenset((a, c)))
            used_pairs.add(frozenset((b, c)))
        
        all_triples = list(combinations(range(N), 3))
        rng = random.Random(42)
        
        for iteration in range(max_iters):
            if time_check():
                break
            
            rng.shuffle(all_triples)
            improved = False
            
            for t in all_triples:
                a, b, c = t
                pairs = frozenset((a, b)), frozenset((a, c)), frozenset((b, c))
                
                # Check linearity
                valid = True
                for p in pairs:
                    if p in used_pairs:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Try adding
                new_triples = current + [list(t)]
                new_alpha = compute_alpha_fast(new_triples)
                
                if new_alpha < current_alpha:
                    current = new_triples
                    current_alpha = new_alpha
                    for p in pairs:
                        used_pairs.add(p)
                    improved = True
                    break
            
            if not improved:
                break
        
        return current

    # Main search
    best_triples = None
    best_alpha = N
    
    # Phase 1: Multiple backtracking constructions with different seeds
    num_constructions = 50
    for i in range(num_constructions):
        if time_check():
            break
        triples = backtracking_construct(i * 7919 + 13)
        if not triples:
            continue
        alpha = compute_alpha_fast(triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples
            # Try local search on improvement
            if best_alpha > 5:
                improved = local_search_add(best_triples, max_iters=50)
                imp_alpha = compute_alpha_fast(improved)
                if imp_alpha < best_alpha:
                    best_alpha = imp_alpha
                    best_triples = improved
        if best_alpha <= 5:
            break
    
    # Phase 2: More constructions with local search
    for i in range(num_constructions, num_constructions + 50):
        if time_check():
            break
        triples = backtracking_construct(i * 104729 + 7)
        if not triples:
            continue
        alpha = compute_alpha_fast(triples)
        if alpha <= best_alpha:
            improved = local_search_add(triples, max_iters=30)
            imp_alpha = compute_alpha_fast(improved)
            if imp_alpha < best_alpha:
                best_alpha = imp_alpha
                best_triples = improved
        if best_alpha <= 5:
            break
    
    if best_triples is None:
        # Fallback: simple construction
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_triples
