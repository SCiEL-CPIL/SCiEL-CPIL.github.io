"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    # All 4-element subsets of [1,16] as bitmasks
    all_sets = []
    for combo in combinations(range(16), 4):
        m = 0
        for x in combo:
            m |= (1 << x)
        all_sets.append(m)
    
    def mask_to_list(m):
        return sorted((i + 1) for i in range(16) if (m >> i) & 1)
    
    # Precompute pairwise intersections for all sets
    # inter[i][j] = all_sets[i] & all_sets[j]
    n_sets = len(all_sets)
    # We'll use a dict-based approach for the family to speed up checking
    
    def greedy_pass(order):
        """Greedy pass: add sets in given order if no sunflower violation."""
        family = []
        family_masks = set()
        
        for C in order:
            if C in family_masks:
                continue
            # Check against all pairs in family
            valid = True
            for i in range(len(family)):
                A = family[i]
                ac = A & C
                for j in range(i + 1, len(family)):
                    B = family[j]
                    ab = A & B
                    bc = B & C
                    if ab == ac == bc:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                family.append(C)
                family_masks.add(C)
        
        return family
    
    def local_search(family, max_time=10.0):
        """Try to improve family by adding sets that fit."""
        start = time.time()
        family = list(family)
        family_masks = set(family)
        available = [s for s in all_sets if s not in family_masks]
        random.shuffle(available)
        
        improved = True
        while improved and (time.time() - start) < max_time:
            improved = False
            for s in available:
                if (time.time() - start) >= max_time:
                    break
                if s in family_masks:
                    continue
                # Check if s can be added
                valid = True
                for i in range(len(family)):
                    A = family[i]
                    ac = A & s
                    for j in range(i + 1, len(family)):
                        B = family[j]
                        ab = A & B
                        bc = B & s
                        if ab == ac == bc:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    family.append(s)
                    family_masks.add(s)
                    available.remove(s)
                    improved = True
        
        return family
    
    def swap_search(family, max_time=10.0):
        """Try to improve family by swapping out one set for another."""
        start = time.time()
        family = list(family)
        family_masks = set(family)
        
        # Precompute which sets are in family
        available = [s for s in all_sets if s not in family_masks]
        random.shuffle(available)
        
        improved = True
        while improved and (time.time() - start) < max_time:
            improved = False
            for s in available:
                if (time.time() - start) >= max_time:
                    break
                if s in family_masks:
                    continue
                # Try removing each family member and adding s
                for idx in range(len(family)):
                    old = family[idx]
                    # Check if (family - {old}) + {s} is valid
                    new_family = family[:idx] + family[idx+1:]
                    # Check if s can be added to new_family
                    valid = True
                    for i in range(len(new_family)):
                        A = new_family[i]
                        ac = A & s
                        for j in range(i + 1, len(new_family)):
                            B = new_family[j]
                            ab = A & B
                            bc = B & s
                            if ab == ac == bc:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        # Perform swap
                        family = new_family + [s]
                        family_masks.discard(old)
                        family_masks.add(s)
                        available.remove(s)
                        improved = True
                        break
                if improved:
                    break
        
        return family
    
    best_family = []
    
    # Multiple greedy restarts with different random orders
    random.seed(42)
    while time.time() < deadline:
        order = all_sets[:]
        random.shuffle(order)
        fam = greedy_pass(order)
        
        if len(fam) > len(best_family):
            best_family = fam
        
        if len(best_family) >= 250:
            break
    
    # Local search to improve best_family
    if time.time() < deadline - 5:
        improved = local_search(best_family, max_time=min(15.0, deadline - time.time() - 5))
        if len(improved) > len(best_family):
            best_family = improved
    
    # Swap search if time remains
    if time.time() < deadline - 5:
        improved = swap_search(best_family, max_time=min(15.0, deadline - time.time() - 5))
        if len(improved) > len(best_family):
            best_family = improved
    
    return [mask_to_list(m) for m in best_family]
# EVOLVE-BLOCK-END
