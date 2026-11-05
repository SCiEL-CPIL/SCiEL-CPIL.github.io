"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = 16
    k = 4
    all_sets = [frozenset(c) for c in combinations(range(1, n+1), k)]
    all_sets_set = set(all_sets)
    
    def is_sunflower(a, b, c):
        """Check if three distinct sets form a sunflower (all pairwise intersections equal)."""
        ab = a & b
        ac = a & c
        bc = b & c
        return ab == ac == bc
    
    def can_add(candidate, family):
        """Check if adding candidate to family creates any sunflower."""
        for i, j in combinations(range(len(family)), 2):
            if is_sunflower(family[i], family[j], candidate):
                return False
        return True
    
    def greedy_build(order, family_size_limit=None):
        """Build family greedily following the given order."""
        family = []
        for s in order:
            if family_size_limit and len(family) >= family_size_limit:
                break
            if can_add(s, family):
                family.append(s)
        return family
    
    def local_improve(family, max_iters=200):
        """Try to improve family by swapping out sets."""
        family = list(family)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Try removing each set and adding a better one
            for i in range(len(family)):
                removed = family[i]
                remaining = [f for j, f in enumerate(family) if j != i]
                # Find a set that can be added to remaining
                for s in all_sets:
                    if s == removed:
                        continue
                    if s in remaining:
                        continue
                    if can_add(s, remaining):
                        # Check if this swap improves anything (same size, so just accept)
                        new_family = remaining + [s]
                        # Prefer to keep diversity - just accept first valid swap
                        family = new_family
                        improved = True
                        break
                if improved:
                    break
        return family
    
    # Strategy 1: Greedy with natural order
    family1 = greedy_build(all_sets)
    best_family = family1
    best_size = len(family1)
    
    # Strategy 2: Multiple random restarts
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        order = all_sets[:]
        random.shuffle(order)
        family = greedy_build(order)
        
        if len(family) > best_size:
            best_family = family
            best_size = len(family)
        
        # Try local improvement on promising families
        if len(family) >= best_size - 2:
            improved = local_improve(family, max_iters=50)
            if len(improved) > best_size:
                best_family = improved
                best_size = len(improved)
    
    # Strategy 3: Greedy with reverse order and other heuristics
    if time.time() < deadline - 5:
        # Try sorting by intersection properties
        def set_key(s):
            # Prefer sets with fewer common elements with others
            return sum(len(s & t) for t in all_sets[:100])
        
        order3 = sorted(all_sets, key=set_key)
        family3 = greedy_build(order3)
        if len(family3) > best_size:
            best_family = family3
            best_size = len(family3)
    
    # Final local improvement on best family
    if time.time() < deadline - 3:
        improved = local_improve(best_family, max_iters=100)
        if len(improved) > best_size:
            best_family = improved
            best_size = len(improved)
    
    # Convert to sorted lists
    result = [sorted(s) for s in best_family]
    return result
# EVOLVE-BLOCK-END
