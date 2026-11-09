"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 80  # 80 second deadline, leaving 10s safety margin
    
    # Generate all possible 4-element subsets of [1,16]
    all_sets = list(combinations(range(1, 17), 4))
    all_set_list = [set(s) for s in all_sets]
    
    # Precompute pairwise intersections for all pairs
    # But that's C(1820, 2) ~ 1.6M pairs, might be too much
    # Instead, we'll compute on the fly but cache
    
    def is_sunflower_triple(a, b, c):
        """Check if three sets form a sunflower (all pairwise intersections equal)"""
        ab = a & b
        ac = a & c
        bc = b & c
        return ab == ac == bc
    
    def check_conflict(new_set, current_family):
        """Check if adding new_set creates a sunflower with any two in current_family"""
        for i in range(len(current_family)):
            for j in range(i+1, len(current_family)):
                if is_sunflower_triple(current_family[i], current_family[j], new_set):
                    return True
        return False
    
    best_result = []
    
    # Try multiple random orderings
    num_trials = 0
    while time.time() < deadline:
        num_trials += 1
        random.shuffle(all_set_list)
        
        current_family = []
        for s in all_set_list:
            if time.time() > deadline - 5:
                break
            if not check_conflict(s, current_family):
                current_family.append(s)
        
        if len(current_family) > len(best_result):
            best_result = current_family[:]
            # If we've found a large family, we can stop early if time is running out
            if len(best_result) > 200 and time.time() > deadline - 10:
                break
    
    return [sorted(s) for s in best_result]
# EVOLVE-BLOCK-END
