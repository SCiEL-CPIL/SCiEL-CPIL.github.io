"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def _is_valid(arr):
    """Check if arr has distinct positive integers and all 2^n subset sums are distinct."""
    n = len(arr)
    # Check distinct positive integers
    if len(set(arr)) != n:
        return False
    if any(x <= 0 for x in arr):
        return False
    # Check all subset sums are distinct
    sums = set()
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += arr[i]
        if s in sums:
            return False
        sums.add(s)
    return True

def _local_search(arr, max_val, time_limit):
    """
    Local search to minimize the maximum element.
    Start from a valid set, try to reduce elements.
    """
    deadline = time.time() + time_limit
    best = list(arr)
    best_max = max(best)
    
    n = len(best)
    
    # Try to reduce each element to a smaller value while maintaining validity
    # We'll do multiple passes
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            old_val = best[i]
            # Try values from 1 up to old_val-1
            # But we need to be smart: try values that don't conflict
            for new_val in range(1, old_val):
                if new_val in best:
                    continue
                candidate = best[:]
                candidate[i] = new_val
                if max(candidate) < max_val and _is_valid(candidate):
                    best = candidate
                    best_max = max(best)
                    improved = True
                    break
    return best, best_max

def _greedy_construct(max_val):
    """
    Greedily construct a set of 12 distinct positive integers <= max_val
    such that all subset sums are distinct.
    Uses a backtracking approach with greedy ordering.
    """
    n = 12
    
    def backtrack(chosen, max_val):
        if len(chosen) == n:
            return chosen[:]
        
        # Determine which values to try
        # Sort candidates: prefer smaller values to keep max small
        used = set(chosen)
        
        # Generate candidate values
        # We want to try values that keep future options open
        # A good heuristic: try values in increasing order, but skip those that cause conflicts
        
        for v in range(1, max_val + 1):
            if v in used:
                continue
            
            # Quick check: does adding v create a duplicate subset sum?
            # Current subset sums
            current_sums = set()
            for mask in range(1 << len(chosen)):
                s = 0
                for i in range(len(chosen)):
                    if mask & (1 << i):
                        s += chosen[i]
                current_sums.add(s)
            
            # New subset sums after adding v
            new_sums = set(current_sums)
            for s in current_sums:
                new_sums.add(s + v)
            
            if len(new_sums) != len(current_sums) * 2:
                continue
            
            # Also check that all elements are distinct (already ensured by used set)
            
            new_chosen = chosen + [v]
            result = backtrack(new_chosen, max_val)
            if result is not None:
                return result
            # Pruning: if we've spent too much time, bail
        return None
    
    return backtrack([], max_val)

def _binary_search_max_val(lo, hi, time_limit):
    """
    Binary search for the smallest feasible max_val.
    """
    deadline = time.time() + time_limit
    best_found = None
    best_max = hi + 1
    
    while lo <= hi and time.time() < deadline - 5:
        mid = (lo + hi) // 2
        # Try to construct with max_val = mid
        result = _greedy_construct(mid)
        if result is not None:
            if max(result) < best_max:
                best_max = max(result)
                best_found = result
            hi = mid - 1
        else:
            lo = mid + 1
    
    return best_found, best_max

def construct():
    deadline = time.time() + 80  # Safety margin
    
    # Known good starting point: powers of 2 give max = 2048
    # But we can do much better.
    # For n=12, the minimum possible max is at least around 2^(n-1) = 1024?
    # Actually, for superincreasing sequences, we need a_i > sum of previous.
    # But we don't need superincreasing, just distinct subset sums.
    
    # A better construction: use a base that's slightly larger than needed.
    # The optimal for n elements with distinct subset sums has max around 2^n / n?
    # Let's try a heuristic construction.
    
    # Start with a reasonable upper bound
    # For n=12, let's try to find a solution with max around 1000-2000
    
    # Strategy: 
    # 1. Try a few known constructions
    # 2. Use local search to improve
    
    # Known: powers of 2 give [1,2,4,...,2048], max=2048
    # Better: use a denser construction
    
    # Let's try a greedy approach with backtracking, starting from a moderate max
    
    # First, try to find a solution with max_val = 1024
    # Then try 1023, 1022, etc.
    
    # Actually, let's use a smarter approach:
    # Start with a valid solution (powers of 2), then locally optimize.
    
    best_arr = [1 << i for i in range(12)]
    best_max = 2048
    
    # Try local search to reduce the maximum
    time_left = deadline - time.time()
    if time_left > 10:
        improved_arr, improved_max = _local_search(best_arr, best_max, time_left - 10)
        if improved_max < best_max:
            best_arr = improved_arr
            best_max = improved_max
    
    # Try different starting points
    time_left = deadline - time.time()
    if time_left > 15:
        # Try a different initial construction
        # Use a base-3 like construction but with smaller numbers
        # Or try random restarts
        
        random.seed(42)
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            
            # Try to construct with a random-ish approach
            # Start with small numbers and grow
            arr = [1]
            for i in range(1, 12):
                # Try to find the smallest value that maintains distinct subset sums
                found = False
                for v in range(1, best_max):
                    if v in arr:
                        continue
                    candidate = arr + [v]
                    if _is_valid(candidate):
                        arr = candidate
                        found = True
                        break
                if not found:
                    break
            else:
                if max(arr) < best_max:
                    best_arr = arr
                    best_max = max(arr)
                
                # Local search from here
                time_left = deadline - time.time()
                if time_left > 5:
                    improved_arr, improved_max = _local_search(best_arr, best_max, time_left - 5)
                    if improved_max < best_max:
                        best_arr = improved_arr
                        best_max = improved_max
    
    # Final check
    if not _is_valid(best_arr):
        # Fallback to powers of 2
        best_arr = [1 << i for i in range(12)]
    
    return best_arr
# EVOLVE-BLOCK-END
