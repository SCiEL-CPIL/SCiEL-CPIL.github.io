"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
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
    
    def can_add(mask_new, chosen_masks):
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

    best_family = []
    best_size = 0
    
    # Strategy 1: Deterministic Greedy (Lexicographic)
    # This matches the inspiration 1 which scored 1.0
    chosen = []
    for mask in all_subsets:
        if time.time() > deadline:
            break
        if can_add(mask, chosen):
            chosen.append(mask)
            
    if len(chosen) > best_size:
        best_size = len(chosen)
        best_family = chosen[:]

    # Strategy 2: Randomized Greedy Restarts
    # Try to find a larger family by changing the order
    # We use a fixed number of restarts, but break early if time is up
    
    # Estimate time per restart
    # If deterministic took T, random might be similar.
    # We have ~85s. If one restart is ~1s, we can do ~80.
    
    num_restarts = 300
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:  # 5s safety margin
            break
        
        # Use a different seed for each restart
        # We can use a simple LCG or just random.seed
        random.seed(12345 + restart)
        shuffled = all_subsets[:]
        random.shuffle(shuffled)
        
        chosen = []
        
        for mask in shuffled:
            if time.time() > deadline - 5.0:
                break
            if can_add(mask, chosen):
                chosen.append(mask)
                
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_family = chosen[:]
            
            # If we hit a very high number, we might stop early?
            # Theoretical max is not known, but 250 is the cap.
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
