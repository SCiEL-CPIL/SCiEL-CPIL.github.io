import time
import itertools
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s
    
    # Helper: check if a set of integers has all distinct reciprocal subset sums
    def is_dissociated(elements):
        """Check if {1/e for e in elements} is dissociated (all subset sums distinct)."""
        n = len(elements)
        if n == 0:
            return True
        # Use Fraction for exact arithmetic
        reciprocals = [Fraction(1, e) for e in elements]
        seen = {Fraction(0, 1)}
        # Build subset sums iteratively
        sums = [Fraction(0, 1)]
        for r in reciprocals:
            new_sums = []
            for s in sums:
                ns = s + r
                if ns in seen:
                    return False
                new_sums.append(ns)
                seen.add(ns)
            sums.extend(new_sums)
        return True
    
    # Greedy construction: try to build a large set
    # Start with all elements, try to find a large valid subset
    
    # Approach: try subsets of decreasing size, starting from large
    # But that's too slow. Instead, use greedy + local search.
    
    # Greedy: add elements one by one if they maintain dissociation
    # Order elements by some heuristic (e.g., smallest first, or largest first)
    
    all_elements = list(range(1, 23))
    
    # Try multiple orders and greedy constructions
    best_set = []
    
    # Order 1: ascending (smallest elements first - larger reciprocals)
    # Order 2: descending
    # Order 3: alternating large/small
    orders = [
        all_elements,
        all_elements[::-1],
        # Alternate: pick large, small, large, small...
        sorted(all_elements, key=lambda x: (x % 2, -x)),
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        current = []
        for e in order:
            if time.time() > deadline:
                break
            candidate = current + [e]
            if is_dissociated(candidate):
                current = candidate
        if len(current) > len(best_set):
            best_set = current[:]
    
    # Try to improve best_set by adding remaining elements
    if time.time() < deadline:
        remaining = [e for e in all_elements if e not in best_set]
        improved = True
        while improved:
            improved = False
            for e in remaining[:]:
                if time.time() > deadline:
                    break
                candidate = best_set + [e]
                if is_dissociated(candidate):
                    best_set = candidate
                    remaining.remove(e)
                    improved = True
    
    # Fallback: use the seed answer if we didn't find anything better
    if not best_set or len(best_set) <= 9:
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    # Final verification
    if not is_dissociated(best_set):
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return best_set
