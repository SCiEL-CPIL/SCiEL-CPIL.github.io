import sys
from math import factorial
from functools import lru_cache
import time

def construct():
    start_time = time.time()
    timeout = 55.0

    target = factorial(12)
    
    # Get all divisors of target
    all_divs = []
    for i in range(1, int(target ** 0.5) + 1):
        if target % i == 0:
            all_divs.append(i)
            if i != target // i:
                all_divs.append(target // i)
    
    # Only divisors > 12
    valid_divs = sorted([d for d in all_divs if d > 12])
    n_divs = len(valid_divs)
    
    if n_divs == 0:
        return [target]
        
    # For each possible max value, we need to check if a solution exists
    # using distinct divisors in (12, M]
    
    # Precompute: for each index, the set of divisors up to that index
    # We'll use binary search on the max index
    
    def check_solution(max_idx):
        """
        Checks if a solution exists using valid_divs[0..max_idx].
        Returns the list of factors if found, else None.
        """
        candidates = valid_divs[:max_idx + 1]
        n = len(candidates)
        
        if n == 0:
            return None
        
        # Create a set for O(1) lookup
        cand_set = set(candidates)
        
        # Precompute suffix products of largest k elements for pruning
        # Since candidates is sorted ascending, the largest k in candidates[i:] 
        # are the last k elements: candidates[n-k], ..., candidates[n-1]
        # But we need to consider only elements from index i onwards.
        # For pruning: max possible product from positions i..n-1 using at most slots_left elements
        # = product of the largest min(slots_left, n-i) elements in candidates[i:]
        
        # Precompute suffix_max_prod[i][k] = product of largest k elements in candidates[i:]
        # If not enough elements, return 0
        suffix_max_prod = [[0] * 13 for _ in range(n + 1)]
        for i in range(n + 1):
            suffix_max_prod[i][0] = 1
        for i in range(n - 1, -1, -1):
            for k in range(1, 13):
                if n - i >= k:
                    # The largest k elements in candidates[i:] are candidates[n-k], ..., candidates[n-1]
                    # But we need them to be >= candidates[i], which they are since sorted ascending
                    # Actually, the largest k in candidates[i:] are the last k elements of the whole array
                    # IF n-k >= i, otherwise it's candidates[i], candidates[i+1], ..., candidates[n-1]
                    if n - k >= i:
                        # Use the last k elements
                        prod = 1
                        for j in range(n - k, n):
                            prod *= candidates[j]
                        suffix_max_prod[i][k] = prod
                    else:
                        # Use all elements from i to n-1
                        prod = 1
                        for j in range(i, n):
                            prod *= candidates[j]
                        suffix_max_prod[i][k] = prod
                else:
                    suffix_max_prod[i][k] = 0
        
        @lru_cache(maxsize=None)
        def dfs(i, rem, count):
            if rem == 1:
                return True
            if i >= n or count >= 12:
                return False
            
            slots_left = 12 - count
            
            # Pruning 1: If rem is greater than the product of the largest slots_left available candidates, fail.
            if suffix_max_prod[i][slots_left] < rem:
                return False
            
            # Pruning 2: If rem is less than the smallest available candidate and we must use at least one more factor
            # Actually, we don't have to use all factors, so this doesn't directly apply.
            # But if rem > 1 and rem is not divisible by any candidate in candidates[i:], we can prune.
            # This is expensive to check, so we'll rely on the loop.
            
            # Pruning 3: If slots_left == 1, check if rem is in candidates[i:]
            if slots_left == 1:
                # Check if rem is in the set and >= candidates[i]
                if rem in cand_set and rem >= candidates[i]:
                    return True
                return False
            
            c = candidates[i]
            
            # Option 1: Take candidates[i] (prefer taking to use more factors, allowing smaller max)
            if rem % c == 0:
                if dfs(i + 1, rem // c, count + 1):
                    return True
            
            # Option 2: Skip candidates[i]
            if dfs(i + 1, rem, count):
                return True
            
            return False

        if dfs(0, target, 0):
            # Reconstruct solution
            result = []
            i, rem, count = 0, target, 0
            while rem != 1 and i < n:
                c = candidates[i]
                slots_left = 12 - count
                
                # Check if we can solve by taking i
                if rem % c == 0 and dfs(i + 1, rem // c, count + 1):
                    # Took
                    result.append(c)
                    rem //= c
                    count += 1
                    i += 1
                elif dfs(i + 1, rem, count):
                    # Skipped
                    i += 1
                else:
                    # Should not happen if dfs returned True
                    break
            
            return result
        else:
            return None

    # Binary search for the minimal M (index in valid_divs)
    low = 0
    high = n_divs - 1
    best_result = None
    
    # Fallback: if binary search fails or times out, return [target]
    # Note: [target] is always a valid solution (1 factor, distinct, >12).
    
    while low <= high:
        if time.time() - start_time > timeout:
            break
        
        mid = (low + high) // 2
        sol = check_solution(mid)
        
        if sol is not None:
            best_result = sol
            high = mid - 1
        else:
            low = mid + 1
    
    if best_result is not None:
        return best_result
    
    # If no solution found in time or via search, return the trivial solution
    # The trivial solution is [12!]
    return [target]

# EVOLVE-BLOCK-END
