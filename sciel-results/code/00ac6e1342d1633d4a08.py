"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s limit to be safe
    
    # Compute LCM of 1..22
    L = reduce(lcm, range(1, 23))
    
    # Weights: w[i] = L // (i+1) for i in 0..21
    # Corresponds to x = 1..22
    weights = [L // (i + 1) for i in range(22)]
    
    # Sort indices by descending weight (ascending x) to prioritize large weights
    # This heuristic often helps in packing/dissociation problems
    indices = list(range(22))
    indices.sort(key=lambda i: weights[i], reverse=True)
    
    # Reorder weights according to sorted indices
    sorted_weights = [weights[i] for i in indices]
    
    best_set = []
    
    def is_valid_addition(current_sums, w):
        """Check if adding weight w to current_sums keeps all sums distinct."""
        new_sums = {s + w for s in current_sums}
        if current_sums & new_sums:
            return False, None
        return True, current_sums | new_sums

    def search(k, start, current_sums, current_indices):
        """
        Try to find a subset of size k from indices[start:]
        that is dissociated, given current_sums and current_indices.
        Returns True if found, and updates best_set.
        """
        if time.time() > deadline:
            return False
            
        if len(current_indices) == k:
            # Found a valid set of size k
            nonlocal best_set
            best_set = current_indices[:]
            return True
            
        # Pruning: not enough elements left
        if 22 - start < k - len(current_indices):
            return False
            
        # Option 1: Take the next element
        w = sorted_weights[start]
        valid, new_sums = is_valid_addition(current_sums, w)
        if valid:
            if search(k, start + 1, new_sums, current_indices + [indices[start]]):
                return True
                
        # Option 2: Skip the next element
        if search(k, start + 1, current_sums, current_indices):
            return True
            
        return False

    # Try to find the maximum cardinality
    # Start from the maximum possible size 22 and go down
    for k in range(22, 0, -1):
        if time.time() > deadline:
            break
        if search(k, 0, {0}, []):
            # Found a set of size k
            break
            
    # If no set found (shouldn't happen, at least size 1 is always valid), return a default
    if not best_set:
        best_set = [indices[0]]
        
    # Convert back to original values (1..22)
    # best_set contains indices from the sorted list, which are the original indices 0..21
    # The value for index i is i+1
    result = [i + 1 for i in best_set]
    
    # Ensure non-empty and sorted for consistency (though order doesn't matter for the set)
    if not result:
        result = [1]
        
    return result
# EVOLVE-BLOCK-END
