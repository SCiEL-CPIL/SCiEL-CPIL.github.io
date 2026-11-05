import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    points = list(range(16))
    all_triples = list(combinations(points, 3))
    n_triples = len(all_triples)  # 560
    
    # Precompute pairs for each triple for fast conflict checking
    triple_pairs = []
    for t in all_triples:
        pairs = frozenset([(t[0], t[1]), (t[0], t[2]), (t[1], t[2])])
        triple_pairs.append(pairs)
    
    # Precompute triple masks for fast independence check
    triple_masks = []
    for t in all_triples:
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.append(mask)
    
    # Map from triple to its index
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    def get_alpha_fast(triple_list):
        """Fast independence number calculation using bitmask enumeration."""
        if not triple_list:
            return 16
        
        # Build a list of triple masks
        masks = [triple_masks[triple_to_idx[t]] for t in triple_list]
        
        # Enumerate all 2^16 subsets, find the largest one with no triple contained
        # For each subset mask, check if any triple is fully contained
        # A triple with mask m is contained in subset s iff (m & s) == m
        
        # We'll iterate by size from 16 down to 1
        # Precompute: for each possible subset mask, which triples are contained?
        # That's expensive. Instead, for each subset, check all triples.
        
        # Better: iterate subsets by size, use combinations
        for k in range(16, 0, -1):
            for subset in combinations(points, k):
                s_mask = 0
                for p in subset:
                    s_mask |= (1 << p)
                
                is_ind = True
                for m in masks:
                    if (m & s_mask) == m:
                        is_ind = False
                        break
                if is_ind:
                    return k
        return 0
    
    def get_alpha_with_set(triple_list):
        """Returns (alpha, independent_set) where independent_set is a set of points."""
        if not triple_list:
            return 16, set(points)
        
        masks = [triple_masks[triple_to_idx[t]] for t in triple_list]
        
        for k in range(16, 0, -1):
            for subset in combinations(points, k):
                s_mask = 0
                for p in subset:
                    s_mask |= (1 << p)
                
                is_ind = True
                for m in masks:
                    if (m & s_mask) == m:
                        is_ind = False
                        break
                if is_ind:
                    return k, set(subset)
        return 0, set()
    
    best_alpha = 16
    best_triples = []
    
    # Helper to check if a triple can be added
    def can_add(triple_idx, used_pairs):
        t_pairs = triple_pairs[triple_idx]
        for p in t_pairs:
            if p in used_pairs:
                return False
        return True
    
    # Strategy: Randomized Greedy Construction with Local Repair
    
    num_restarts = 5000
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # 1. Construct an initial linear system
        indices = list(range(n_triples))
        random.shuffle(indices)
        
        current_triples = []
        used_pairs = set()
        
        for idx in indices:
            if can_add(idx, used_pairs):
                t = all_triples[idx]
                current_triples.append(t)
                for p in triple_pairs[idx]:
                    used_pairs.add(p)
                
        # Initial alpha
        alpha = get_alpha_fast(current_triples)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = list(current_triples)
        
        # Try to improve locally
        improved = True
        while improved and time.time() < deadline:
            improved = False
            alpha, indep_set = get_alpha_with_set(current_triples)
            
            if alpha <= best_alpha:
                break
                
            # Find all triples contained in indep_set that can be added
            candidates = []
            for idx in range(n_triples):
                t = all_triples[idx]
                if all(p in indep_set for p in t):
                    if can_add(idx, used_pairs):
                        candidates.append(idx)
                        
            if candidates:
                # Pick a random candidate
                idx = random.choice(candidates)
                t = all_triples[idx]
                current_triples.append(t)
                for p in triple_pairs[idx]:
                    used_pairs.add(p)
                improved = True
            else:
                # No triple can be added inside the current independent set.
                # Try a perturbation: remove a random triple and re-add
                if current_triples and random.random() < 0.3:
                    rem_pos = random.randrange(len(current_triples))
                    t_rem = current_triples.pop(rem_pos)
                    rem_idx = triple_to_idx[t_rem]
                    for p in triple_pairs[rem_idx]:
                        used_pairs.discard(p)
                    # Try to add a different triple
                    new_candidates = []
                    for idx2 in range(n_triples):
                        if can_add(idx2, used_pairs):
                            new_candidates.append(idx2)
                    if new_candidates:
                        idx2 = random.choice(new_candidates)
                        t2 = all_triples[idx2]
                        current_triples.append(t2)
                        for p in triple_pairs[idx2]:
                            used_pairs.add(p)
                        improved = True

        # After local search, check if we improved best
        alpha = get_alpha_fast(current_triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = list(current_triples)

    return best_triples
