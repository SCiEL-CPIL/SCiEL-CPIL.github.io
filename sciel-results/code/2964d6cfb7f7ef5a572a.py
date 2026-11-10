"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations
from math import gcd
from functools import reduce

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    start_time = time.time()
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    def check_dissociated(elements):
        """Check if all subset sums of reciprocals are distinct using integer arithmetic."""
        k = len(elements)
        if k == 0:
            return True
        
        # Compute LCM of all elements
        L = reduce(lcm, elements)
        
        # Convert reciprocals to integers: 1/x -> L/x
        int_vals = [L // x for x in elements]
        
        # Generate all subset sums and check for duplicates
        sums = {0}
        for v in int_vals:
            new_sums = set()
            for s in sums:
                ns = s + v
                if ns in sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        
        return len(sums) == (1 << k)
    
    def check_dissociated_fast(elements):
        """Faster check using incremental subset sum tracking."""
        k = len(elements)
        if k == 0:
            return True
        if k > 16:  # Too many subsets to track efficiently
            return check_dissociated(elements)
        
        L = reduce(lcm, elements)
        int_vals = [L // x for x in elements]
        
        # Use bitset-like approach for small k
        sums = {0}
        for v in int_vals:
            new_sums = set()
            for s in sums:
                ns = s + v
                if ns in sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        
        return len(sums) == (1 << k)
    
    # Precompute conflict pairs: (a,b) where 1/a + 1/b = 1/c for some c
    # This means a*b = c*(a+b), so c = a*b/(a+b) must be integer
    # Also check 1/a = 1/b + 1/c => a*b = c*(a+b) same condition
    # And 1/a + 1/b + 1/c = 1/d etc. - but for pairwise conflicts in the set,
    # we mainly care about when adding x to set S, does any subset sum collision occur?
    
    # Better: precompute for each pair (a,b) whether they can coexist
    # Two elements a,b conflict if 1/a + 1/b equals 1/c for some c, or 1/a = 1/b + 1/c, etc.
    # But the real check is: when we add element x to set S, do all 2^|S|+2^(|S|+1) subset sums remain distinct?
    
    # For efficiency, let's use a different approach:
    # A set is dissociated iff no nonempty subset has sum 0 in the reciprocal sense
    # i.e., no subset T1, T2 with T1 != T2 have sum(1/x for x in T1) = sum(1/x for x in T2)
    # This is equivalent to: all 2^k subset sums are distinct
    
    # Greedy construction with multiple orderings
    candidates = list(range(1, 23))
    
    best_set = []
    best_size = 0
    
    # Generate more diverse orderings
    orderings = [
        sorted(candidates, reverse=True),  # Largest first
        sorted(candidates),                 # Smallest first
        [22, 1, 21, 2, 20, 3, 19, 4, 18, 5, 17, 6, 16, 7, 15, 8, 14, 9, 13, 10, 12, 11],
        [2, 3, 5, 7, 11, 13, 17, 19, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22],
        [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 2, 3, 5, 7, 11, 13, 17, 19],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        [22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        [1, 2, 4, 8, 16, 3, 6, 12, 5, 10, 20, 7, 14, 21, 9, 18, 11, 22, 13, 15, 17, 19],
        # Add more strategic orderings
        [22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        [16, 12, 8, 4, 2, 20, 15, 10, 5, 22, 11, 6, 3, 18, 9, 7, 14, 21, 13, 19, 17, 1],
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1],
    ]
    
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
            if check_dissociated_fast(test_set):
                current = test_set
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
    
    # Branch-and-bound local search: try to add elements in order of "conflict-ness"
    if best_set and time.time() < deadline - 5:
        # Sort remaining elements by how "compatible" they seem with current set
        # Heuristic: elements with larger values tend to be more compatible
        remaining = [x for x in candidates if x not in best_set]
        remaining.sort(reverse=True)
        
        # Try adding each remaining element
        for elem in remaining:
            if time.time() > deadline - 5:
                break
            test_set = best_set + [elem]
            if check_dissociated_fast(test_set):
                best_set = test_set
                best_size = len(best_set)
        
        # Now try swaps: replace one element with another to potentially enable more additions
        for _ in range(3):  # Few rounds of swap improvement
            if time.time() > deadline - 5:
                break
            progress = False
            for i, elem in enumerate(best_set):
                if time.time() > deadline - 5:
                    break
                for new_elem in candidates:
                    if new_elem in best_set:
                        continue
                    candidate_set = best_set[:i] + [new_elem] + best_set[i+1:]
                    candidate_set.sort()
                    if check_dissociated_fast(candidate_set):
                        # Check if this swap enables adding more elements
                        best_set = candidate_set
                        progress = True
                        # Try to add more after swap
                        for add_elem in candidates:
                            if time.time() > deadline - 5:
                                break
                            if add_elem in best_set:
                                continue
                            test_set = best_set + [add_elem]
                            if check_dissociated_fast(test_set):
                                best_set = test_set
                        break
                if progress:
                    break
            if not progress:
                break
    
    # Final polish: try all possible additions one more time
    if time.time() < deadline - 3:
        for elem in sorted(candidates, reverse=True):
            if time.time() > deadline - 3:
                break
            if elem in best_set:
                continue
            test_set = best_set + [elem]
            if check_dissociated_fast(test_set):
                best_set = test_set
    
    if not best_set:
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return best_set
# EVOLVE-BLOCK-END
