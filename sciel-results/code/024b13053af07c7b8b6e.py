"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from functools import reduce

def _compute_lcm_1_22():
    return reduce(lcm, range(1, 23))

_L = _compute_lcm_1_22()

def _is_dissociated(subset):
    """Check if all subset sums of reciprocals (scaled by LCM) are distinct."""
    k = len(subset)
    if k == 0:
        return True
    vals = [_L // a for a in subset]
    sums = {0}
    for v in vals:
        new_sums = set()
        for s in sums:
            ns = s + v
            if ns in sums or ns in new_sums:
                return False
            new_sums.add(ns)
        sums.update(new_sums)
    return True

def _backtrack_search(deadline):
    """
    Backtracking search to find maximum cardinality dissociated subset of [1..22].
    Uses exact integer arithmetic via LCM.
    Returns the best subset found (max cardinality, then lexicographically smallest for determinism).
    """
    best = []
    n = 22
    vals = [_L // i for i in range(1, n + 1)]
    
    # We search for the largest valid subset.
    # Strategy: try to include elements in increasing order, backtracking when collision detected.
    # To maximize cardinality, we prefer adding smaller numbers (more "room" in the sum space? 
    # Actually, smaller numbers create smaller increments, potentially more collisions? 
    # Let's just do a standard DFS that tries to build the largest set.
    # Order: try to add elements 1..22. If we can't add an element, skip it.
    # But to find the MAXIMUM, we should try all combinations? No, that's 2^22 which is ~4M.
    # 4M is fast enough if the check is fast.
    # However, the check itself is O(2^k).
    # For k=10, 2^10=1024. 4M * 1024 is too slow.
    
    # Better: Incremental backtracking.
    # Keep a set of current sums. When adding a new element v, check if any new sum collides with existing or other new sums.
    
    def dfs(idx, current_sums, current_subset):
        nonlocal best
        if time.time() > deadline:
            return
        
        if len(current_subset) > len(best):
            best = current_subset[:]
        
        # Pruning: if we can't beat the best size even by adding all remaining elements
        remaining = n - idx
        if len(current_subset) + remaining <= len(best):
            return
        
        for i in range(idx, n):
            if time.time() > deadline:
                return
            
            v = vals[i]
            
            # Check if adding v to current_sums creates any duplicate
            # New sums are {s + v for s in current_sums}
            # Collision if (s1 + v) == (s2 + v) => s1 == s2 (impossible since current_sums is a set)
            # Collision if (s1 + v) == s2 for some s1, s2 in current_sums
            # i.e., v == s2 - s1.
            
            # Efficient check:
            # Compute new_sums = {s + v for s in current_sums}
            # If len(new_sums) != len(current_sums), there's a collision within new sums? No, s1+v == s2+v => s1==s2.
            # Collision is if new_sums intersects current_sums.
            
            new_sums = set()
            valid = True
            for s in current_sums:
                ns = s + v
                if ns in current_sums:
                    valid = False
                    break
                new_sums.add(ns)
            
            if valid:
                # Also need to check if any two new sums are equal? 
                # s1 + v == s2 + v => s1 == s2. Since s1, s2 distinct in current_sums, new sums are distinct from each other.
                # So we just need new_sums to be disjoint from current_sums.
                
                new_total_sums = current_sums.union(new_sums)
                dfs(i + 1, new_total_sums, current_subset + [i + 1])
            
            # Note: We don't backtrack by "removing" from a global structure, we pass new sets.
            # This is clean but might be slow due to set copying.
            # For n=22, depth is max 22. Sets size max 2^22? No, valid sets are dissociated, so size is bounded.
            # Actually, if it's dissociated, all 2^k sums are distinct.
            
    dfs(0, {0}, [])
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Start with a known good set or empty
    # The seed has 9 elements.
    # Let's run the backtracking search.
    # To speed up, we can order the search to find large sets quickly.
    # Trying smaller numbers first might lead to larger sets?
    # Let's just run the DFS.
    
    # Optimization: If we find a set of size 10, we know max is at least 10.
    # We can prune branches that can't exceed current best.
    
    best_subset = _backtrack_search(deadline)
    
    # Fallback if search was too slow or failed
    if not best_subset:
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return best_subset

# EVOLVE-BLOCK-END
