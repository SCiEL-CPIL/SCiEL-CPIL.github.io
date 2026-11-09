"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_sunflower_triple(a, b, c):
    """Check if three sets form a sunflower (all pairwise intersections equal)."""
    return a & b == a & c == b & c

def construct():
    deadline = time.time() + 85  # Safety margin before 90s limit
    
    # Pre-compute all 4-element subsets of [1,16]
    all_sets = [set(row) for row in combinations(range(1, 17), 4)]
    n_total = len(all_sets)
    
    # Greedy construction with multiple random restarts
    best_family = []
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Randomly shuffle the candidate sets
        candidates = all_sets[:]
        random.shuffle(candidates)
        
        chosen = []
        for s in candidates:
            if time.time() > deadline:
                break
            
            # Check if adding s creates a sunflower triple with any two existing sets
            can_add = True
            for i in range(len(chosen)):
                if not can_add:
                    break
                for j in range(i + 1, len(chosen)):
                    if _is_sunflower_triple(chosen[i], chosen[j], s):
                        can_add = False
                        break
            
            if can_add:
                chosen.append(s)
        
        if len(chosen) > len(best_family):
            best_family = chosen[:]
        
        # If we've found a large family, we can stop early
        if len(best_family) >= 200:
            break
    
    return [sorted(s) for s in best_family]
# EVOLVE-BLOCK-END
