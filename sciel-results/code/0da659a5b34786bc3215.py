"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill

    n = 16
    k = 4

    # Precompute all 4-subsets as bitmasks
    all_subsets = []
    for row in combinations(range(n), k):
        mask = 0
        for x in row:
            mask |= (1 << x)
        all_subsets.append(mask)

    num_subsets = len(all_subsets)

    def can_add_fast(mask_new, chosen_masks):
        # Group chosen sets by their intersection with mask_new
        # For a sunflower with M, A, B: A∩M = B∩M = A∩B = I
        # So we group by I = A∩M, and within each group check if any pair has intersection I
        groups = defaultdict(list)
        for m in chosen_masks:
            inter = m & mask_new
            groups[inter].append(m)
        
        for inter, group in groups.items():
            if len(group) < 2:
                continue
            L = len(group)
            for i in range(L):
                gi = group[i]
                for j in range(i + 1, L):
                    gj = group[j]
                    if (gi & gj) == inter:
                        return False
        return True

    def local_search(chosen_masks):
        # Try to improve the family by adding sets that fit, then removing one and trying alternatives
        # This is a simple 1-move local search
        chosen = set(chosen_masks)
        improved = True
        while improved:
            if time.time() > deadline - 2.0:
                break
            improved = False
            # Try adding each unchosen set
            for mask in all_subsets:
                if mask in chosen:
                    continue
                if can_add_fast(mask, list(chosen)):
                    chosen.add(mask)
                    improved = True
                    break
            # Try 1-swap: remove one set, add another
            if not improved:
                for mask_remove in list(chosen):
                    if time.time() > deadline - 2.0:
                        break
                    chosen.discard(mask_remove)
                    for mask_add in all_subsets:
                        if mask_add in chosen:
                            continue
                        if can_add_fast(mask_add, list(chosen)):
                            chosen.add(mask_add)
                            improved = True
                            break
                    if not improved:
                        chosen.add(mask_remove)  # put it back
        return list(chosen)

    best_family = []
    best_size = 0

    # Strategy 1: Deterministic Greedy (Lexicographic)
    chosen = []
    for mask in all_subsets:
        if time.time() > deadline - 5.0:
            break
        if can_add_fast(mask, chosen):
            chosen.append(mask)
    
    if len(chosen) > best_size:
        best_size = len(chosen)
        best_family = chosen[:]

    # Local search on the greedy result
    if time.time() < deadline - 10.0:
        improved = local_search(best_family)
        if len(improved) > best_size:
            best_size = len(improved)
            best_family = improved

    # Strategy 2: Randomized Greedy Restarts with Local Search
    num_restarts = 200
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:  # 10s safety margin for local search
            break
        
        # Use a different seed for each restart
        random.seed(12345 + restart * 7)
        shuffled = all_subsets[:]
        random.shuffle(shuffled)
        
        chosen = []
        
        for mask in shuffled:
            if time.time() > deadline - 10.0:
                break
            if can_add_fast(mask, chosen):
                chosen.append(mask)
                
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_family = chosen[:]
            
            # If we hit a very high number, we might stop early?
            # Theoretical max is not known, but 250 is the cap.
            if best_size >= 250:
                break
        
        # Local search on the random greedy result
        if time.time() < deadline - 5.0 and len(chosen) >= best_size - 5:
            improved = local_search(chosen)
            if len(improved) > best_size:
                best_size = len(improved)
                best_family = improved
                if best_size >= 250:
                    break

    # Convert back to sorted lists
    result = []
    for mask in best_family:
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
    
    return result
# EVOLVE-BLOCK-END
