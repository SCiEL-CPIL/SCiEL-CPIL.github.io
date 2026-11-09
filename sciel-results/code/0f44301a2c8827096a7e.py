import time
import random
from itertools import combinations
from math import lcm

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s margin
    
    # Precompute conflicts
    # conflicts[x] is a list of frozensets {a, b} such that {x, a, b} is a bad triple
    # We only need a < b to avoid duplicates, but for subset check, order doesn't matter.
    # We'll store as frozenset to allow set operations.
    conflicts = [None] * 129
    for x in range(1, 129):
        conf = []
        for a in range(1, 129):
            if a == x:
                continue
            for b in range(a + 1, 129):
                if b == x:
                    continue
                # Check if {x, a, b} is a bad triple
                # lcm(x,a) == lcm(x,b) == lcm(a,b)
                if lcm(x, a) == lcm(x, b) == lcm(a, b):
                    conf.append(frozenset((a, b)))
        conflicts[x] = conf

    def is_valid_addition(x, S):
        """Check if x can be added to set S without forming a bad triple."""
        # S is a set of integers
        # We need to check if any conflict pair of x is a subset of S
        for pair in conflicts[x]:
            if pair.issubset(S):
                return False
        return True

    def greedy_construct(order):
        S = set()
        for x in order:
            if is_valid_addition(x, S):
                S.add(x)
        return S

    def local_search(S):
        """Try to improve S by adding elements or swapping."""
        S = set(S)
        improved = True
        while improved:
            improved = False
            # Try to add
            for x in range(1, 129):
                if x not in S:
                    if is_valid_addition(x, S):
                        S.add(x)
                        improved = True
                        break # Restart loop to check for more additions
            
            if not improved:
                # Try to swap
                for x in range(1, 129):
                    if x not in S:
                        # Find all bad pairs for x in S
                        bad_pairs = [p for p in conflicts[x] if p.issubset(S)]
                        if not bad_pairs:
                            # Should have been caught in add phase, but just in case
                            S.add(x)
                            improved = True
                            break
                        
                        # Find intersection of all bad pairs
                        # If intersection is non-empty, we can remove an element from it
                        # Intersection of sets
                        inter = set(bad_pairs[0])
                        for p in bad_pairs[1:]:
                            inter.intersection_update(p)
                            if not inter:
                                break
                        
                        if inter:
                            # We can swap out any y in inter
                            # Pick one arbitrarily (e.g., first one)
                            y = next(iter(inter))
                            S.remove(y)
                            S.add(x)
                            improved = True
                            break
        
        return S

    best_set = set()
    best_size = 0
    
    # Run multiple random greedy passes
    while time.time() < deadline:
        order = list(range(1, 129))
        random.shuffle(order)
        
        S = greedy_construct(order)
        S = local_search(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    return list(best_set)

# EVOLVE-BLOCK-END
