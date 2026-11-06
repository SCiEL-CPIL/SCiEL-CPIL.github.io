"""Rewritten program for the suite task; optimized for speed and search depth."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations
from collections import defaultdict

def construct():
    n = 16
    k = 4
    
    # Generate all 4-element subsets as bitmasks
    all_masks = []
    for comb in combinations(range(n), k):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        all_masks.append(mask)
    
    # Theoretical upper bound from Erdos-Rado Sunflower Lemma
    # Max size without 3-sunflower is at most k!(k-1)^2 = 24 * 9 = 216
    upper_bound = 216
    
    best_family = []
    start_time = time.time()
    time_limit = 50.0  # Keep 10s margin for 60s limit
    
    # We will perform multiple randomized greedy restarts
    while time.time() - start_time < time_limit:
        random.shuffle(all_masks)
        family = []
        family_set = set() # For fast membership if needed, though we use list for iteration
        
        # Greedy construction
        for s in all_masks:
            if s in family_set:
                continue
                
            # Check if adding s creates a sunflower with any pair in family
            conflict = False
            
            # Group existing sets in family by their intersection with s
            # Intersection I = A & s
            groups = defaultdict(list)
            for a in family:
                inter = a & s
                groups[inter].append(a)
                
            # Check for conflicts within each group
            # If A, B are in group I, then A & s = I and B & s = I.
            # Sunflower condition: A & B = I.
            for I, members in groups.items():
                if len(members) < 2:
                    continue
                # Check all pairs in this group
                for i in range(len(members)):
                    a = members[i]
                    for j in range(i + 1, len(members)):
                        b = members[j]
                        if (a & b) == I:
                            conflict = True
                            break
                    if conflict:
                        break
                if conflict:
                    break
            
            if not conflict:
                family.append(s)
                family_set.add(s)
                
                # If we hit the theoretical max, we can stop this restart
                if len(family) >= upper_bound:
                    break
        
        # Update best family
        if len(family) > len(best_family):
            best_family = family
            
            # If we hit the theoretical max, we can stop all restarts
            if len(best_family) >= upper_bound:
                break
        
    # Convert bitmasks back to sorted lists of integers
    result = []
    for mask in best_family:
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(i + 1)
        result.append(subset)
        
    return result
# EVOLVE-BLOCK-END
