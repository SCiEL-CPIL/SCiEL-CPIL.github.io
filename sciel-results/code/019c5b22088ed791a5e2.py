# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 55.0
    
    N = 22
    # Reciprocals for each number 1..22
    recips = [Fraction(1, i) for i in range(1, N + 1)]
    
    best_set = None
    best_size = 0
    
    # Try to find largest dissociated set
    # Use backtracking: try to include elements, check dissociation
    # Order: try larger numbers first (smaller reciprocals) as they're "safer"
    # Actually, let's try a greedy + backtracking hybrid
    
    def is_dissociated(elements):
        """Check if the set of reciprocals for given elements is dissociated."""
        if not elements:
            return True
        # Generate all subset sums
        sums = set()
        sums.add(Fraction(0))
        for i in elements:
            r = recips[i - 1]
            new_sums = set()
            for s in sums:
                ns = s + r
                if ns in sums or ns in new_sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        return True
    
    # Backtracking search
    # Try all sizes from 14 down to 1
    # For each size, try combinations
    
    # Better: use recursive backtracking with pruning
    def backtrack(available, current, depth, max_depth):
        nonlocal best_set, best_size
        if time.time() - start_time > timeout:
            return
        if len(current) > best_size:
            best_set = current.copy()
            best_size = len(current)
        if len(current) == max_depth:
            return
        if not available:
            return
        
        for idx in range(len(available)):
            elem = available[idx]
            new_current = current + [elem]
            # Quick check: is this still dissociated?
            if is_dissociated(new_current):
                # Recurse
                remaining = available[idx + 1:]
                backtrack(remaining, new_current, depth + 1, max_depth)
            # Also try skipping this element
            # (handled by continuing loop)
    
    # Try different orderings
    # Order 1: decreasing (larger numbers first)
    order1 = list(range(N, 0, -1))
    backtrack(order1, [], 0, 15)
    
    if time.time() - start_time > timeout:
        if best_set:
            return best_set
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    # Order 2: increasing
    order2 = list(range(1, N + 1))
    backtrack(order2, [], 0, 15)
    
    if best_set:
        return best_set
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
# EVOLVE-BLOCK-END
