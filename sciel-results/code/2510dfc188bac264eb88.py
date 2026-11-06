import time
import random
from math import lcm
from functools import reduce

def construct():
    """
    Find a nonempty subset of [1,22] with maximum cardinality such that
    all reciprocal subset sums (including empty sum 0) are distinct.
    """
    # Deadline guard: 85 seconds to be safe against 90s kill
    start_time = time.time()
    deadline = start_time + 85.0
    
    def time_left():
        return deadline - time.time()
    
    # Compute LCM of 1..22 to convert reciprocal sums to integer sums
    L = reduce(lcm, range(1, 23))
    
    # Map each element a to its integer contribution L // a
    vals = {a: L // a for a in range(1, 23)}
    
    # Elements sorted by value descending.
    # Larger values are more "constraining" as they create larger sums
    # and are more likely to cause collisions.
    elements = sorted(range(1, 23), key=lambda a: vals[a], reverse=True)
    
    # Precompute values in this sorted order for faster access
    sorted_vals = [vals[a] for a in elements]
    
    def is_dissociated_fast(current_sums, v):
        """
        Check if adding value v to the set of current sums results in
        a set with distinct sums.
        Uses set intersection for O(n) check instead of O(n^2).
        """
        for s in current_sums:
            if s + v in current_sums:
                return False
        return True
    
    def greedy_build(order=None):
        """
        Greedily build a dissociated set by trying elements in order.
        Returns a valid dissociated subset.
        """
        if order is None:
            order = elements
            
        current_sums = {0}
        result = []
        for a in order:
            v = vals[a]
            if is_dissociated_fast(current_sums, v):
                new_sums = current_sums | {s + v for s in current_sums}
                current_sums = new_sums
                result.append(a)
        return result
    
    # First, get a greedy solution as a baseline using the standard descending order
    greedy_result = greedy_build()
    best_result = greedy_result
    best_size = len(greedy_result)
    
    # Try some other deterministic orderings
    if time_left() > 5:
        # Ascending order
        asc_order = sorted(range(1, 23), key=lambda a: vals[a])
        asc_result = greedy_build(asc_order)
        if len(asc_result) > best_size:
            best_result = asc_result
            best_size = len(asc_result)
        
        # Random orderings
        random.seed(42)
        for _ in range(50):
            if time_left() < 5:
                break
            rand_order = elements[:]
            random.shuffle(rand_order)
            rand_result = greedy_build(rand_order)
            if len(rand_result) > best_size:
                best_result = rand_result
                best_size = len(rand_result)
    
    def find_subset_of_size(target_size):
        """
        Use backtracking to find a subset of exactly target_size elements
        from the sorted elements list that is dissociated.
        Returns the subset (list of original element values) or None.
        """
        n = len(elements)
        
        def backtrack(idx, current_size, current_sums, current_subset):
            if time_left() < 2:
                return None
            
            # Pruning 1: Not enough elements left to reach target_size
            if current_size + (n - idx) < target_size:
                return None
            
            # Base case: We found a valid subset of the required size
            if current_size == target_size:
                return list(current_subset)
            
            # Try to include elements from idx to end
            for i in range(idx, n):
                if time_left() < 2:
                    return None
                
                a = elements[i]
                v = sorted_vals[i]
                
                # Check if adding v is valid
                if is_dissociated_fast(current_sums, v):
                    # Calculate new sums
                    new_sums = current_sums | {s + v for s in current_sums}
                    
                    # Recurse
                    current_subset.append(a)
                    result = backtrack(i + 1, current_size + 1, new_sums, current_subset)
                    current_subset.pop()
                    
                    if result is not None:
                        return result
            
            return None

        return backtrack(0, 0, {0}, [])

    # Search for the maximum cardinality
    # We try sizes from 22 down to best_size + 1 (since best_size is our current baseline)
    # As soon as we find a valid subset of size T, T is the maximum.
    for size in range(22, best_size, -1):
        if time_left() < 10:
            break
            
        result = find_subset_of_size(size)
        if result is not None:
            return result
        
        # If we didn't find one, continue to smaller sizes
    
    # Return the best result we found (greedy baseline or better)
    return best_result if best_result else [1]
