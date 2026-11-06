"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill

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

    def is_compatible(c, selected):
        """
        Check if candidate c can be added to selected without forming a sunflower.
        A sunflower is a triple {A, B, C} such that A∩B = B∩C = A∩C.
        So for any pair (A, B) in selected, we must NOT have:
        c∩A == A∩B AND c∩B == A∩B
        """
        n_sel = len(selected)
        # Optimization: if selected is large, this is the bottleneck.
        # We check all pairs in selected.
        for i in range(n_sel):
            a = selected[i]
            ca = c & a
            # Precompute a&b for j > i? No, just compute on fly.
            for j in range(i + 1, n_sel):
                b = selected[j]
                ab = a & b
                if ca == ab and (c & b) == ab:
                    return False
        return True

    best_family = []
    best_size = 0

    # Strategy: Multiple Randomized Greedy Restarts
    # Each restart is fast, allowing many attempts.
    max_restarts = 100
    for restart in range(max_restarts):
        if time.time() > deadline - 2.0:
            break
        
        # Shuffle candidates randomly
        shuffled = all_subsets[:]
        random.shuffle(shuffled)
        
        chosen = []
        
        for mask in shuffled:
            if time.time() > deadline - 2.0:
                break
            
            # Check compatibility
            if is_compatible(mask, chosen):
                chosen.append(mask)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_family = chosen[:]
            
            # If we find a very good solution, we can try to polish it with local search
            # But local search is expensive. Let's just do more restarts first.
            if best_size >= 200:
                # Try a few more restarts to see if we can beat it
                continue

    # If time remains, try a simple local search on the best family found
    # Local search: try to add any missing set that fits.
    # This is O(N * K^2) which might be too slow if K is large.
    # Let's only do it if K is small enough or time is ample.
    if time.time() < deadline - 5.0 and best_size < 250:
        current_best = best_family[:]
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for mask in all_subsets:
                if mask in current_best:
                    continue
                if is_compatible(mask, current_best):
                    current_best.append(mask)
                    improved = True
                    break # Restart loop to check new additions
            
            if len(current_best) > best_size:
                best_size = len(current_best)
                best_family = current_best[:]

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
