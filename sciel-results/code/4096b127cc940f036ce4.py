"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations
from math import gcd
from functools import reduce
import random

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    start_time = time.time()
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    def check_dissociated_fast(elements):
        """Check if all subset sums of reciprocals are distinct using integer arithmetic."""
        k = len(elements)
        if k == 0:
            return True
        if k > 20:
            # Too many to check quickly with this method
            return False
        
        # Compute LCM of all elements
        L = reduce(lcm, elements)
        
        # Convert reciprocals to integers: 1/x -> L/x
        int_vals = [L // x for x in elements]
        
        # Generate all subset sums and check for duplicates
        sums = set()
        sums.add(0)
        for v in int_vals:
            new_sums = set()
            for s in sums:
                ns = s + v
                if ns in sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        
        return len(sums) == (1 << k)
    
    def check_dissociated(elements):
        """Check if all subset sums of reciprocals are distinct."""
        k = len(elements)
        if k == 0:
            return True
        if k > 18:
            # Use Fraction for larger sets, but be careful about performance
            reciprocals = [Fraction(1, x) for x in elements]
            sums = set()
            sums.add(Fraction(0, 1))
            for r in reciprocals:
                new_sums = set()
                for s in sums:
                    ns = s + r
                    if ns in sums:
                        return False
                    new_sums.add(ns)
                sums.update(new_sums)
            return len(sums) == (1 << k)
        else:
            return check_dissociated_fast(elements)
    
    # Greedy construction: try to build a large dissociated set
    candidates = list(range(1, 23))
    
    best_set = []
    best_size = 0
    
    # Try different orderings to find a large set
    # Strategy: prefer larger denominators (smaller reciprocals) as they're less likely to cause collisions
    orderings = [
        sorted(candidates, reverse=True),  # Largest first
        sorted(candidates),                 # Smallest first
        # Interleaved: large, small, medium, etc.
        [22, 1, 21, 2, 20, 3, 19, 4, 18, 5, 17, 6, 16, 7, 15, 8, 14, 9, 13, 10, 12, 11],
        # Primes first
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22],
        # Composite-heavy
        [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 2, 3, 5, 7, 11, 13, 17, 19],
        # Alternating parity
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        # Reverse alternating
        [22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    ]
    
    # Add randomized orderings for more diversity
    rng = random.Random(42)
    for _ in range(30):
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        orderings.append(shuffled)
    
    for order in orderings:
        if time.time() > deadline:
            break
        
        current = []
        for elem in order:
            if time.time() > deadline:
                break
            if elem in current:
                continue
            test_set = current + [elem]
            if check_dissociated(test_set):
                current = test_set
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
    
    # Local improvement: try swapping elements to potentially increase size
    if best_set and time.time() < deadline - 5:
        for _ in range(5):  # Try more rounds of local improvement
            if time.time() > deadline - 5:
                break
            progress = False
            for i, elem in enumerate(best_set):
                if time.time() > deadline - 5:
                    break
                # Try replacing elem with a candidate not in best_set
                for new_elem in candidates:
                    if new_elem in best_set:
                        continue
                    candidate_set = best_set[:i] + [new_elem] + best_set[i+1:]
                    candidate_set.sort()
                    if check_dissociated(candidate_set):
                        best_set = candidate_set
                        progress = True
                        break
                if progress:
                    break
            if not progress:
                break
    
    # Final attempt: try to add more elements to best_set in multiple orders
    if time.time() < deadline - 5:
        for add_order in [sorted(candidates, reverse=True), sorted(candidates)]:
            if time.time() > deadline - 5:
                break
            for elem in add_order:
                if time.time() > deadline - 5:
                    break
                if elem in best_set:
                    continue
                test_set = best_set + [elem]
                if check_dissociated(test_set):
                    best_set = test_set
    
    if not best_set:
        # Fallback to known good set
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return best_set
# EVOLVE-BLOCK-END
