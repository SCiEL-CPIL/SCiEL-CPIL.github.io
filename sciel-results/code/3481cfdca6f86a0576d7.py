import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Generate all 4-subsets of [1, 16] as bitmasks
    candidates = []
    for comb in combinations(range(1, 17), 4):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        candidates.append(mask)
    
    N = len(candidates)  # 1820
    
    # Precompute pairwise intersections
    # Using a list of lists for O(1) access
    # pair_int[i][j] = candidates[i] & candidates[j]
    pair_int = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            val = candidates[i] & candidates[j]
            pair_int[i][j] = val
            pair_int[j][i] = val

    def can_add(family, k):
        """
        Check if candidate k can be added to family without forming a sunflower.
        family: list of indices
        k: index
        """
        n_fam = len(family)
        for a in range(n_fam):
            idx_a = family[a]
            # Optimization: if idx_a == k, skip (shouldn't happen)
            for b in range(a + 1, n_fam):
                idx_b = family[b]
                # Check if (idx_a, idx_b, k) is a sunflower
                # Condition: A∩B == A∩C == B∩C
                kab = pair_int[idx_a][idx_b]
                kak = candidates[idx_a] & candidates[k]
                if kab != kak:
                    continue
                kbk = candidates[idx_b] & candidates[k]
                if kab == kbk:
                    return False
        return True

    best_family = []
    best_size = 0
    
    # Multi-start greedy
    # We can run many fast greedy passes
    attempts = 0
    while time.time() < deadline - 2.0:
        attempts += 1
        random.seed(attempts)
        order = list(range(N))
        random.shuffle(order)
        
        family = []
        fam_set = set()
        
        for idx in order:
            if idx in fam_set:
                continue
            
            if can_add(family, idx):
                family.append(idx)
                fam_set.add(idx)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family.copy()
            
        # If we found a very large family, we might try local search?
        # For now, stick to fast greedy restarts as they are reliable and fast.
        # The previous SA was likely too slow.
        
        # Optional: Local search on best_family
        # If time permits, try to improve best_family
        if best_size > 0 and time.time() < deadline - 5.0:
             # Simple local search: try to swap out one element to allow two new ones?
             # Or just try to add more to best_family?
             # Let's try a greedy pass on the remaining candidates for the current best
             # But we need to be careful not to break the "no sunflower" property.
             # The `can_add` function ensures this.
             
             # Let's try to grow best_family with a different random order
             # But best_family is a list. We need to convert to set for fast lookup if we do multiple passes.
             # Actually, the greedy above already does one pass.
             # Doing multiple passes with different shuffles on the *same* growing family might help.
             
             # Let's try a "second phase" greedy on the best_family found so far
             # This is risky if not careful, but `can_add` is safe.
             
             current_best = best_family.copy()
             current_best_set = set(current_best)
             
             # Try adding any remaining candidate that fits
             # Sort candidates by some heuristic? Random is fine.
             remaining = [i for i in range(N) if i not in current_best_set]
             random.shuffle(remaining)
             
             improved = False
             for idx in remaining:
                 if can_add(current_best, idx):
                     current_best.append(idx)
                     current_best_set.add(idx)
                     improved = True
             
             if improved and len(current_best) > best_size:
                 best_size = len(current_best)
                 best_family = current_best

    # Convert best_family (list of indices) to set of frozensets
    result = set()
    for idx in best_family:
        mask = candidates[idx]
        s = frozenset(i for i in range(1, 17) if (mask >> i) & 1)
        result.add(s)
        
    return result
