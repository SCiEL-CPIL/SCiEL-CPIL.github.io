import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 55.0
    
    N = 22
    candidates = list(range(1, N + 1))
    
    # Cache for LCM computations
    lcm_cache = {}
    
    def get_lcm(S):
        """Get LCM of a tuple with caching."""
        key = tuple(sorted(S))
        if key in lcm_cache:
            return lcm_cache[key]
        L = 1
        for x in S:
            L = math.lcm(L, x)
        lcm_cache[key] = L
        return L
    
    def is_dissociated(S):
        """
        Check if the set S is dissociated, i.e., all subset sums of reciprocals are distinct.
        Equivalent to checking if all subset sums of {L/x for x in S} are distinct,
        where L = lcm(S).
        """
        if not S:
            return True
        
        # Calculate LCM
        L = get_lcm(S)
        
        # Convert to integer values
        vals = [L // x for x in S]
        
        # Sort vals in descending order to detect collisions earlier
        # Larger values create larger gaps, making collisions less likely to hide
        vals_sorted = sorted(vals, reverse=True)
        
        # Generate all subset sums with early exit
        sums = {0}
        for i, v in enumerate(vals_sorted):
            new_sums = set()
            for s in sums:
                new_sums.add(s + v)
            sums.update(new_sums)
            # Check for collision
            # Expected size is 2^(i+1)
            if len(sums) < (1 << (i + 1)):
                return False
                
        return len(sums) == (1 << len(S))

    def backtrack(current, available, target_size, time_limit):
        if time.time() - start_time > time_limit:
            return None
            
        if len(current) == target_size:
            return current
            
        if len(current) + len(available) < target_size:
            return None
            
        for i, x in enumerate(available):
            new_current = current + [x]
            
            # Check if new_current is dissociated
            if is_dissociated(new_current):
                result = backtrack(new_current, available[i+1:], target_size, time_limit)
                if result is not None:
                    return result
            else:
                continue
                
        return None

    # Try to find the largest possible set
    # Start from N down to 1
    for k in range(N, 0, -1):
        # Try to find a set of size k
        # Use descending order: larger numbers first
        # Larger numbers have smaller reciprocals, which might be "easier" to pack
        available = sorted(candidates, reverse=True)
        
        result = backtrack([], available, k, timeout)
        
        if result is not None:
            return result
            
        # If time is running out, break and return best found
        if time.time() - start_time > timeout - 5:
             break

    # Fallback: Return a known valid set
    # {1, 2, 4, 8, 16} is valid (powers of 2)
    return [1, 2, 4, 8, 16]
