import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n = 16
    all_triples = list(combinations(range(n), 3))
    
    # Precompute pairs for each triple
    triple_pairs = []
    for t in all_triples:
        a, b, c = t
        triple_pairs.append({(a, b) if a < b else (b, a), 
                             (a, c) if a < c else (c, a), 
                             (b, c) if b < c else (c, b)})

    def check_time():
        return time.time() > deadline

    def compute_alpha(triples):
        if not triples:
            return n
            
        triple_masks = []
        for t in triples:
            mask = 0
            for x in t:
                mask |= (1 << x)
            triple_masks.append(mask)
        
        nt = len(triple_masks)
        
        for k in range(n, 0, -1):
            if check_time():
                return n
            found = False
            for subset in combinations(range(n), k):
                mask = 0
                for x in subset:
                    mask |= (1 << x)
                
                is_indep = True
                for tm in triple_masks:
                    if (mask & tm) == tm:
                        is_indep = False
                        break
                if is_indep:
                    return k
        return 0

    def greedy_fill(used_pairs, seed):
        """
        Greedily add triples to the system, given current used_pairs.
        Returns list of added triple indices.
        """
        rng = random.Random(seed)
        indices = list(range(len(all_triples)))
        rng.shuffle(indices)
        
        added = []
        for idx in indices:
            if check_time():
                break
            pairs = triple_pairs[idx]
            if all(p not in used_pairs for p in pairs):
                added.append(idx)
                for p in pairs:
                    used_pairs.add(p)
        return added

    def random_packing(seed):
        used_pairs = set()
        packing = greedy_fill(used_pairs, seed)
        return packing

    def perturb_and_refill(packing, k, seed):
        """
        Remove k random triples from packing, then refill greedily.
        """
        if len(packing) < k:
            return packing
            
        rng = random.Random(seed)
        # Pick k indices to remove
        remove_indices = set(rng.sample(packing, k))
        
        # Remove them from used_pairs
        used_pairs = set()
        new_packing = []
        for idx in packing:
            if idx in remove_indices:
                continue
            new_packing.append(idx)
            for p in triple_pairs[idx]:
                used_pairs.add(p)
                
        # Refill
        added = greedy_fill(used_pairs, seed + 1)
        new_packing.extend(added)
        
        return new_packing

    best_packing = None
    best_alpha = n + 1
    
    # Phase 1: Random Restarts
    num_restarts = 100
    for i in range(num_restarts):
        if check_time():
            break
        p = random_packing(i)
        if not p:
            continue
        a = compute_alpha(p)
        if a < best_alpha:
            best_alpha = a
            best_packing = p
            if best_alpha <= 4:
                break

    # Phase 2: Local Search (Perturb and Refill)
    if best_packing:
        # Try to improve best_packing
        # Use multiple iterations
        for ls_iter in range(50):
            if check_time():
                break
            
            # Try different k values for removal
            # k=1 is impossible to improve size, but might change structure?
            # No, if we remove 1, we can only add 1 (same one).
            # So k must be >= 2 to allow different triples?
            # If we remove 2 triples (6 pairs), we can add 2 different triples using those 6 pairs?
            # Yes.
            
            improved = False
            for k in [2, 3, 4]:
                if check_time():
                    break
                # Try several random perturbations
                for attempt in range(10):
                    if check_time():
                        break
                    new_p = perturb_and_refill(best_packing, k, i * 100 + ls_iter * 10 + attempt)
                    new_a = compute_alpha(new_p)
                    
                    if new_a < best_alpha:
                        best_alpha = new_a
                        best_packing = new_p
                        improved = True
                        break # Restart with new best
                    elif new_a == best_alpha:
                        # Keep it to explore?
                        # Let's accept it to allow further perturbations
                        # But don't mark as improved to avoid infinite loop on same alpha
                        # Actually, let's just keep the first one found with same alpha
                        # to change the "landscape"
                        if not improved:
                            best_packing = new_p
                            # Don't break, try other k?
                            # Let's break to restart the loop with the new packing
                            break
                if improved:
                    break

    if not best_packing:
        return [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [0, 3, 6]]

    return [all_triples[idx] for idx in best_packing]
