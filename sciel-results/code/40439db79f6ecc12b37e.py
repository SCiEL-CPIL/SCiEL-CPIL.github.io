"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s safety margin
    
    # Generate all 4-element subsets of [1, 16] as bitmasks
    all_masks = []
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_masks.append(mask)
    
    def is_sunflower(a, b, c):
        # Check if three sets form a sunflower (all pairwise intersections equal)
        ab = a & b
        ac = a & c
        bc = b & c
        return ab == ac == bc
    
    def greedy_random(seed=42):
        random.seed(seed)
        indices = list(range(len(all_masks)))
        random.shuffle(indices)
        chosen = []
        for idx in indices:
            m = all_masks[idx]
            conflict = False
            for i, a in enumerate(chosen):
                for b in chosen[i+1:]:
                    if is_sunflower(a, b, m):
                        conflict = True
                        break
                if conflict:
                    break
            if not conflict:
                chosen.append(m)
            if time.time() > deadline:
                break
        return chosen
    
    # Try a few random greedy runs and keep the best
    best_family = []
    for seed in range(10):
        if time.time() > deadline:
            break
        family = greedy_random(seed)
        if len(family) > len(best_family):
            best_family = family
    
    # Convert bitmasks back to sorted lists of 1-based integers
    result = []
    for mask in best_family:
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
    
    return result
# EVOLVE-BLOCK-END
