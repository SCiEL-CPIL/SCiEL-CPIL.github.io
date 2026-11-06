"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction
from math import gcd
from functools import reduce
from collections import deque

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Compute LCM of all denominators
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, denominators)
    
    # Precompute L/d for each denominator
    L_over_d = [L // d for d in denominators]
    
    deadline = time.time() + 55  # 5s safety margin
    
    # BFS over achievable deficit values
    # deficit = L - sum(L/d_i) for i in S
    # We want smallest positive deficit
    
    # State: (deficit_value, frozenset_of_indices)
    # But that's too large. Instead, track just the deficit and reconstruct.
    
    # Better approach: BFS by adding elements, tracking deficit
    # Start from empty set: deficit = L
    # Add element i: deficit -= L_over_d[i]
    # We want to find smallest positive deficit
    
    # Use a priority queue / best-first search on deficit
    # State: (deficit, tuple_of_sorted_indices)
    
    best_deficit = L  # Start with empty set
    best_indices = tuple()
    
    # Priority queue: (deficit, indices_tuple)
    # We'll use a simple approach: explore states in order of deficit
    
    visited = set()  # Set of index tuples we've seen
    
    # Start with empty set
    queue = deque([(L, ())])
    visited.add(())
    
    while queue and time.time() < deadline:
        current_deficit, current_indices = queue.popleft()
        
        if current_deficit <= 0:
            continue
        
        if current_deficit < best_deficit:
            best_deficit = current_deficit
            best_indices = current_indices
        
        # Try adding each element not already in current_indices
        for i in range(n):
            if i in current_indices:
                continue
            
            new_deficit = current_deficit - L_over_d[i]
            
            # We need new_deficit > 0 (strictly below 1)
            if new_deficit <= 0:
                continue
            
            new_indices = current_indices + (i,) if not current_indices else tuple(sorted(current_indices + (i,)))
            
            # Actually, let's keep indices as a sorted tuple for canonical form
            idx_set = set(current_indices)
            idx_set.add(i)
            new_indices = tuple(sorted(idx_set))
            
            if new_indices in visited:
                continue
            
            visited.add(new_indices)
            queue.append((new_deficit, new_indices))
    
    # If BFS didn't find anything, fall back to greedy
    if not best_indices:
        best_indices_set = set()
        best_sum_frac = Fraction(0)
        for i in range(n):
            if best_sum_frac + Fraction(1, denominators[i]) < 1:
                best_indices_set.add(i)
                best_sum_frac += Fraction(1, denominators[i])
        best_indices = tuple(sorted(best_indices_set))
    
    return list(best_indices)
# EVOLVE-BLOCK-END
