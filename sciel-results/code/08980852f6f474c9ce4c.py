"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 85  # Safety margin before 90s limit
    
    # Precompute all 4-subsets of [1,16]
    all_sets = [set(c) for c in combinations(range(1, 17), 4)]
    
    # Sunflower check: three distinct sets A, B, C form a sunflower if
    # |A ∩ B| = |A ∩ C| = |B ∩ C| (all three pairwise intersections equal)
    # This includes the empty-core case where all pairwise intersections are empty.
    
    def is_sunflower(a, b, c):
        """Check if three 4-sets form a sunflower (all pairwise intersections equal)."""
        ab = len(a & b)
        ac = len(a & c)
        bc = len(b & c)
        return ab == ac == bc
    
    # Greedy construction with randomization and restarts
    best_family = []
    
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        # Shuffle the candidate sets for this attempt
        candidates = all_sets[:]
        random.shuffle(candidates)
        
        family = []
        for s in candidates:
            if time.time() > deadline:
                break
            
            # Check against all pairs in current family
            valid = True
            for i in range(len(family)):
                for j in range(i + 1, len(family)):
                    if is_sunflower(family[i], family[j], s):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                family.append(s)
        
        if len(family) > len(best_family):
            best_family = family[:]
            if len(best_family) >= 250:
                break
    
    # Return the best family found
    return [sorted(s) for s in best_family]
# EVOLVE-BLOCK-END
