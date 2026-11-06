"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from math import gcd
from functools import reduce
from itertools import combinations

def lcm(a, b):
    return a * b // gcd(a, b)

def lcm_list(lst):
    return reduce(lcm, lst, 1)

def construct():
    """
    Find a nonempty subset of [1, 22] such that ALL reciprocal subset sums are distinct.
    Maximize cardinality.
    """
    
    # We search for the maximum possible size K.
    # The maximum size is bounded by 22.
    # We try K from 12 down to 1.
    # For each K, we use backtracking to see if a valid set exists.
    
    def max_dissociated_size(max_n=22, target_k=12):
        """
        Use backtracking to find a dissociated set of size target_k from [1, max_n].
        Returns the set if found, None otherwise.
        """
        # Candidates are 1..max_n
        # We try to build the set element by element.
        # To prune, we can sort candidates.
        # Which order? 
        # If we pick small numbers early, the LCM grows fast, and the "granularity" of the sumset becomes fine, making it easier to avoid collisions with larger numbers?
        # Or does it make it harder?
        # If LCM is large, weights are small integers.
        # Let's try increasing order first, as small numbers are the most "dangerous" for collisions (large weights).
        
        candidates = list(range(1, max_n + 1))
        
        # State for backtracking
        # current_set: list of chosen numbers
        # current_lcm: LCM of chosen numbers
        # current_sums: set of subset sums scaled by current_lcm
        
        # We want to find ANY set of size target_k.
        # We can use a DFS.
        
        import sys
        sys.setrecursionlimit(10000)
        
        def dfs(idx, current_set, current_lcm, current_sums):
            if len(current_set) == target_k:
                return current_set
            
            # Pruning: if remaining candidates + current size < target_k, fail
            if len(current_set) + (max_n - idx + 1) < target_k:
                return None
            
            for i in range(idx, max_n + 1):
                n = candidates[i]
                
                # Calculate new LCM
                new_lcm = lcm(current_lcm, n)
                
                # Check for overflow? LCM of [1..22] is huge but fits in Python int.
                
                # Calculate new sums
                # Old sums are scaled by new_lcm / current_lcm
                scale = new_lcm // current_lcm
                new_weight = new_lcm // n
                
                new_sums = set()
                # Add old sums (scaled)
                for s in current_sums:
                    new_sums.add(s * scale)
                
                # Add new sums (old scaled + new_weight)
                collision = False
                for s in current_sums:
                    ns = s * scale + new_weight
                    if ns in new_sums:
                        collision = True
                        break
                    new_sums.add(ns)
                
                if not collision:
                    res = dfs(i + 1, current_set + [n], new_lcm, new_sums)
                    if res:
                        return res
            
            return None
        
        # Start DFS
        # Initial state: empty set
        # LCM = 1, Sums = {0}
        return dfs(0, [], 1, {0})

    # Try to find the largest K
    # We know K=9 is possible. Let's try 12, 11, 10...
    # Note: C(22, 12) is large, but backtracking prunes heavily if collisions are frequent.
    # If we order candidates by increasing value, we hit collisions early for small sets.
    # Maybe we should try to find a set of size 11 or 12.
    
    # Let's try K from 12 down to 1
    for k in range(12, 0, -1):
        result = max_dissociated_size(22, k)
        if result:
            return sorted(result)
            
    # Fallback if something goes wrong (should not happen as k=1 is always valid)
    return [1]

# EVOLVE-BLOCK-END
