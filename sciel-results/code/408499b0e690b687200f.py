"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # Compute LCM of 1..22 for exact integer arithmetic
    L = reduce(lcm, range(1, 23))
    
    # We want to find the maximum cardinality subset of [1,22] such that
    # all reciprocal subset sums are distinct.
    # Multiply by L: for subset S, values are L//a for a in S.
    # All subset sums of these values must be distinct.
    # This is equivalent to: the set {L//a : a in S} is "dissociated"
    # (all subset sums distinct), which means no two different subsets have the same sum.
    
    # Strategy: Backtracking search from largest cardinality downward.
    # For each candidate size k, try to find a valid subset of size k.
    
    # Precompute the integer values
    vals = {a: L // a for a in range(1, 23)}
    elements = list(range(1, 23))
    
    def is_dissociated(subset):
        """Check if all subset sums of vals[a] for a in subset are distinct."""
        k = len(subset)
        if k == 0:
            return True
        v_list = [vals[a] for a in subset]
        # Generate all 2^k subset sums and check for duplicates
        sums = set()
        sums.add(0)
        for v in v_list:
            new_sums = set()
            for s in sums:
                ns = s + v
                if ns in sums or ns in new_sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        return True
    
    # Try from largest possible size (22) down to 1
    best_result = [1]  # fallback
    
    for k in range(22, 0, -1):
        if time.time() > deadline - 5:
            break
        
        # Try to find a valid subset of size k using backtracking
        # We'll use a recursive backtracking with pruning
        
        def backtrack(remaining_elements, chosen, current_sums_set):
            """
            remaining_elements: list of elements we can still choose from
            chosen: list of currently chosen elements
            current_sums_set: set of all subset sums of chosen elements (as integers)
            Returns a valid subset of size k or None.
            """
            if time.time() > deadline - 3:
                return None
            
            needed = k - len(chosen)
            if needed == 0:
                return chosen[:]
            
            if len(remaining_elements) < needed:
                return None
            
            # Try each element in remaining_elements
            for i, elem in enumerate(remaining_elements):
                v = vals[elem]
                
                # Check if adding v creates a duplicate subset sum
                # New sums would be: existing sums + {s + v for s in existing sums}
                # We need all (s + v) to be distinct from each other and from existing sums
                new_sums = set()
                valid = True
                for s in current_sums_set:
                    ns = s + v
                    if ns in current_sums_set or ns in new_sums:
                        valid = False
                        break
                    new_sums.add(ns)
                
                if not valid:
                    continue
                
                # Recurse with remaining elements after i
                result = backtrack(remaining_elements[i+1:], chosen + [elem], current_sums_set | new_sums)
                if result is not None:
                    return result
            
            return None
        
        # Start backtracking for size k
        # Order elements by some heuristic - try smaller values first or larger?
        # Let's try in natural order first
        result = backtrack(elements, [], {0})
        
        if result is not None:
            best_result = sorted(result)
            break  # Found maximum cardinality, we're done
    
    return best_result
# EVOLVE-BLOCK-END
