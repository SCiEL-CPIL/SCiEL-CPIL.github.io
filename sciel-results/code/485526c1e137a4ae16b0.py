"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s safety margin before 60s limit
    
    # Generate all 4-element subsets of [1,16] as bitmasks (0-indexed bits for 0..15)
    all_masks = []
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_masks.append(mask)
    
    n_total = len(all_masks)  # 1820
    
    def is_sunflower(a, b, c):
        # Check if three sets form a sunflower (all pairwise intersections equal)
        ab = a & b
        ac = a & c
        bc = b & c
        return ab == ac == bc
    
    def greedy_random(seed=42):
        random.seed(seed)
        indices = list(range(n_total))
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
                return chosen
        return chosen
    
    # Try multiple random greedy runs and keep the best
    best_family = []
    best_size = 0
    
    for seed in range(100):
        if time.time() > deadline:
            break
        family = greedy_random(seed)
        if len(family) > best_size:
            best_size = len(family)
            best_family = family
    
    # If we haven't found anything good yet, try a deterministic approach
    if best_size < 50:
        # Try the original deterministic approach
        chosen = []
        for row in combinations(range(1, 17), 4):
            s = set(row)
            valid = True
            for a, b in combinations(chosen, 2):
                if (a & b) == (a & s) == (b & s):
                    valid = False
                    break
            if valid:
                chosen.append(s)
            if time.time() > deadline:
                break
        
        if len(chosen) > best_size:
            best_family = []
            for s in chosen:
                mask = 0
                for elem in s:
                    mask |= (1 << (elem - 1))
                best_family.append(mask)
            best_size = len(chosen)
    
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
