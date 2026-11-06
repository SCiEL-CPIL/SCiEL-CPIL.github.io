import time
from math import lcm
from functools import reduce

def construct():
    """
    Find a nonempty subset of [1,22] with maximum cardinality such that
    all reciprocal subset sums (including empty sum 0) are distinct.
    """
    # Deadline guard: 80 seconds to be safe against 90s kill
    deadline = time.time() + 80
    
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
        # Check if any s + v is in current_sums
        # Equivalent to: (current_sums + v) & current_sums == empty set
        # We can check by seeing if v is in the difference set
        # Or simply: does (current_sums + v) intersect current_sums?
        # This is equivalent to: exists s1, s2 in current_sums such that s1 + v = s2
        # i.e., v = s2 - s1 for some s1, s2 in current_sums
        # So we need to check if v is in {s2 - s1 : s1, s2 in current_sums}
        # But that's expensive. Simpler: for each s in current_sums, check if s+v in current_sums
        # But we can early exit.
        for s in current_sums:
            if s + v in current_sums:
                return False
        return True

    def greedy_build():
        """
        Greedily build a dissociated set by trying elements in order.
        Returns a valid dissociated subset.
        """
        current_sums = {0}
        result = []
        for i, a in enumerate(elements):
            v = sorted_vals[i]
            if is_dissociated_fast(current_sums, v):
                new_sums = current_sums | {s + v for s in current_sums}
                current_sums = new_sums
                result.append(a)
        return result

    # First, get a greedy solution as a baseline
    greedy_result = greedy_build()
    greedy_size = len(greedy_result)
    
    def find_subset_of_size(target_size):
        """
        Use backtracking to find a subset of exactly target_size elements
        from the sorted elements list that is dissociated.
        Returns the subset (list of original element values) or None.
        """
        def backtrack(idx, current_size, current_sums, current_subset):
            if time.time() > deadline:
                return None
            
            # Pruning 1: Not enough elements left to reach target_size
            if current_size + (len(elements) - idx) < target_size:
                return None
            
            # Base case: We found a valid subset of the required size
            if current_size == target_size:
                return list(current_subset)
            
            # Try to include elements from idx to end
            for i in range(idx, len(elements)):
                if time.time() > deadline:
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
    # We try sizes from 22 down to greedy_size (since greedy gives us a baseline)
    # As soon as we find a valid subset of size T, T is the maximum.
    best_result = greedy_result
    
    for size in range(22, greedy_size, -1):
        if time.time() > deadline - 2:
            break
            
        result = find_subset_of_size(size)
        if result is not None:
            return result
        
        # If we didn't find one, continue to smaller sizes
    
    # Return the best result we found (greedy baseline or better)
    return best_result if best_result else [1]
