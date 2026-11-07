"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import sys
import time

def construct():
    """
    Find a large set of distinct integers in 1..256 such that:
    1. All nonempty subset sums are distinct.
    2. No two distinct subset sums divide one another.
    
    Uses backtracking with incremental subset sum maintenance and divisibility checks.
    """
    
    # State: current_set (list), current_sums (set of ints)
    # We assume current_sums contains all non-empty subset sums of current_set.
    
    def check_addition(current_sums, x):
        """
        Check if adding x to the set keeps it valid.
        Returns the new set of sums if valid, else None.
        """
        new_sums = set(current_sums)
        # Add x
        new_sums.add(x)
        # Add s + x for all s in current_sums
        for s in current_sums:
            new_sums.add(s + x)
        
        # Check distinctness: len(new_sums) should be len(current_sums) * 2 + 1?
        # No, current_sums has 2^k - 1 elements.
        # New sums: x, and s+x.
        # If x is not in current_sums and no s+x collides with existing or other new sums, it's distinct.
        # Since we used a set, collisions are merged.
        # Expected size: len(current_sums) + 1 + len(current_sums) = 2*len(current_sums) + 1.
        if len(new_sums) != 2 * len(current_sums) + 1:
            return None
        
        # Check divisibility
        # Sort the new sums to make checking easier?
        # Or just iterate.
        sums_list = list(new_sums)
        n = len(sums_list)
        # Optimization: Only check pairs where a < b.
        # We can sort and then for each a, check b % a == 0.
        sums_list.sort()
        for i in range(n):
            a = sums_list[i]
            # a cannot divide itself (distinct sums)
            # Check if a divides any b > a
            # If a is large, there are few multiples.
            # If a is small, there are many.
            # Simple O(N^2) check might be too slow for N=1000 (10^6 ops).
            # But N grows as 2^k. k=10 -> N=1023. 1023^2 ~ 10^6.
            # In Python, 10^6 modulo ops is ~0.1s.
            # Backtracking depth might be 10-15.
            # We need to be fast.
            
            # Better check:
            # For each a in sums_list, we don't need to check all b.
            # We can precompute? No, sums change.
            
            # Let's try a simple loop first. If it's too slow, we optimize.
            # Actually, we can prune: if a > sums_list[-1] // 2, then 2*a > max_sum, so a cannot divide any b in the set (except itself).
            max_sum = sums_list[-1]
            if a > max_sum // 2:
                continue
                
            # Check multiples of a
            # Multiples: 2a, 3a, ... <= max_sum
            # We need to check if any of these are in new_sums.
            # Using a set for membership is O(1).
            for multiple in range(2 * a, max_sum + 1, a):
                if multiple in new_sums:
                    return None
        
        return new_sums

    def solve(timeout=55):
        start_time = time.time()
        best_set = []
        
        # Candidates: 1..256
        # Strategy: Try to add large numbers first?
        # Large numbers create large sums, which are less likely to be divisors of other sums?
        # Or small numbers?
        # Let's try descending order: 256, 255, ...
        # Or maybe a mix.
        
        candidates = list(range(256, 0, -1))
        
        # Recursive backtracking
        def backtrack(current_set, current_sums, idx):
            nonlocal best_set, start_time
            
            if time.time() - start_time > timeout:
                return
            
            if len(current_set) > len(best_set):
                best_set = list(current_set)
            
            if idx >= len(candidates):
                return
            
            # Pruning: Even if we add all remaining, can we beat best?
            if len(current_set) + (len(candidates) - idx) <= len(best_set):
                return
            
            # Try each candidate from idx onwards
            for i in range(idx, len(candidates)):
                cand = candidates[i]
                
                # Try adding cand
                new_sums = check_addition(current_sums, cand)
                if new_sums is not None:
                    current_set.append(cand)
                    backtrack(current_set, new_sums, i + 1)
                    current_set.pop()
                    
                    # Optional: If we found a very large set, we can stop early?
                    # No, we want the max.
        
        backtrack([], set(), 0)
        return best_set

    result = solve()
    
    if not result:
        # Fallback to a known valid small set
        # {3, 5} is valid.
        # {3, 5, 17} was invalid.
        # Let's just return [3, 5] if search fails completely.
        result = [3, 5]
        
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
