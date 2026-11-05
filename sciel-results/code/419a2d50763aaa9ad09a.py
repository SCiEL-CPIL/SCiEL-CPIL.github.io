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
    
    # Map from triple to index
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute conflict graph: two triples conflict if they share 2+ points
    conflict_masks = [0] * n_triples
    for i in range(n_triples):
        for j in range(i+1, n_triples):
            common = triple_masks[i] & triple_masks[j]
            if bin(common).count('1') >= 2:
                conflict_masks[i] |= (1 << j)
                conflict_masks[j] |= (1 << i)
    
    def compute_covered(triple_list):
        """Compute bitmask of all points that appear in at least one triple."""
        covered = 0
        for t in triple_list:
            covered |= triple_masks[triple_to_idx[t]]
        return covered
    
    def get_alpha_fast(triple_list):
        """Fast alpha computation using bitmask operations on all 2^16 subsets."""
        if not triple_list:
            return 16
        
        # Build a set of triple masks for fast lookup
        tmask_set = set(triple_masks[triple_to_idx[t]] for t in triple_list)
        
        # For each subset S of points, check if any triple is fully contained in S
        # A subset S is independent if no triple mask is a subset of S's mask
        # We check all 2^16 subsets and find the largest independent one
        
        best = 0
        for s_mask in range(1 << 16):
            # Quick check: if s_mask has fewer bits than best, skip
            if bin(s_mask).count('1') <= best:
                continue
            
            # Check if any triple is fully contained in s_mask
            is_ind = True
            for tm in tmask_set:
                if (tm & s_mask) == tm:
                    is_ind = False
                    break
            
            if is_ind:
                best = bin(s_mask).count('1')
        
        return best
    
    def get_alpha_and_set(triple_list):
        """Find largest independent set size and the set itself."""
        if not triple_list:
            return 16, set(points)
        
        tmask_set = set(triple_masks[triple_to_idx[t]] for t in triple_list)
        
        best_size = 0
        best_mask = 0
        
        for s_mask in range(1 << 16):
            cnt = bin(s_mask).count('1')
            if cnt <= best_size:
                continue
            
            is_ind = True
            for tm in tmask_set:
                if (tm & s_mask) == tm:
                    is_ind = False
                    break
            
            if is_ind:
                best_size = cnt
                best_mask = s_mask
        
        # Convert best_mask to set of points
        indep_set = set()
        for p in range(16):
            if best_mask & (1 << p):
                indep_set.add(p)
        
        return best_size, indep_set
    
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
