import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    points = list(range(16))
    all_triples = list(combinations(points, 3))
    n_triples = len(all_triples)
    
    # Precompute bitmask for each triple
    triple_masks = []
    for t in all_triples:
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.append(mask)
    
    # Precompute conflict graph: two triples conflict if they share 2+ points
    # Build adjacency as bitmasks for speed
    conflict_masks = [0] * n_triples
    for i in range(n_triples):
        for j in range(i+1, n_triples):
            # Check if triples share 2+ points
            common = triple_masks[i] & triple_masks[j]
            if bin(common).count('1') >= 2:
                conflict_masks[i] |= (1 << j)
                conflict_masks[j] |= (1 << i)
    
    def get_alpha_and_set(triple_list):
        """Find largest independent set size and the set itself."""
        if not triple_list:
            return 16, set(points)
        
        masks = [triple_masks[all_triples.index(t)] for t in triple_list]
        
        for k in range(16, 0, -1):
            for subset in combinations(points, k):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                
                is_ind = True
                for m in masks:
                    if (m & mask) == m:
                        is_ind = False
                        break
                if is_ind:
                    return k, set(subset)
        return 0, set()
    
    def get_alpha_fast(triple_list):
        """Fast alpha computation using bitmask operations."""
        if not triple_list:
            return 16
        
        masks = [triple_masks[all_triples.index(t)] for t in triple_list]
        
        for k in range(16, 0, -1):
            for subset in combinations(points, k):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                
                is_ind = True
                for m in masks:
                    if (m & mask) == m:
                        is_ind = False
                        break
                if is_ind:
                    return k
        return 0
    
    best_alpha = 16
    best_triples = []
    
    # Strategy: Greedy construction with alpha-minimizing triple selection
    # At each step, pick the triple that most reduces the independence number
    
    num_restarts = 5000
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random initial ordering for diversity
        indices = list(range(n_triples))
        random.shuffle(indices)
        
        current_triples = []
        used_conflicts = set()  # Set of triple indices that conflict with current
        
        # Phase 1: Greedy construction - pick triples that maximize alpha reduction
        for idx in indices:
            if idx in used_conflicts:
                continue
            
            # Check if adding this triple would reduce alpha
            test_triples = current_triples + [all_triples[idx]]
            new_alpha = get_alpha_fast(test_triples)
            
            # Only add if it doesn't increase alpha (shouldn't happen) and we can fit
            if new_alpha <= 16:
                current_triples.append(all_triples[idx])
                # Mark all conflicting triples as used
                cm = conflict_masks[idx]
                while cm:
                    j = (cm & -cm).bit_length() - 1
                    used_conflicts.add(j)
                    cm ^= (1 << j)
        
        # Phase 2: Local search - add triples within the largest independent set
        alpha, indep_set = get_alpha_and_set(current_triples)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            alpha, indep_set = get_alpha_and_set(current_triples)
            
            if alpha <= best_alpha:
                break
            
            # Find all triples contained in indep_set that can be added
            candidates = []
            for idx in range(n_triples):
                if idx in used_conflicts:
                    continue
                t = all_triples[idx]
                if all(p in indep_set for p in t):
                    candidates.append(idx)
            
            if candidates:
                # Pick the candidate that most reduces alpha
                best_idx = None
                best_new_alpha = alpha
                for idx in candidates:
                    test_triples = current_triples + [all_triples[idx]]
                    new_alpha = get_alpha_fast(test_triples)
                    if new_alpha < best_new_alpha:
                        best_new_alpha = new_alpha
                        best_idx = idx
                
                if best_idx is not None:
                    current_triples.append(all_triples[best_idx])
                    cm = conflict_masks[best_idx]
                    while cm:
                        j = (cm & -cm).bit_length() - 1
                        used_conflicts.add(j)
                        cm ^= (1 << j)
                    improved = True
            else:
                # No more triples can be added in the independent set
                # Try removing a random triple and re-adding
                if current_triples and time.time() < deadline - 5:
                    rem_pos = random.randrange(len(current_triples))
                    t_rem = current_triples.pop(rem_pos)
                    rem_idx = all_triples.index(t_rem)
                    
                    # Remove from used_conflicts
                    # This is complex; just restart this local search
                    current_triples.append(t_rem)
                break
        
        # Update best
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = current_triples[:]
            
            # If we've achieved a good alpha, we can try more restarts
            if best_alpha <= 7:
                break
    
    return best_triples
