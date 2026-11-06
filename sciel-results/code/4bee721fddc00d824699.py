import time
from math import factorial
from functools import lru_cache

def construct():
    deadline = time.time() + 80  # safety margin
    target = factorial(12)  # 479001600
    
    # 1. Generate all divisors of target
    # 12! = 2^10 * 3^5 * 5^2 * 7^1
    divisors = set()
    for a in range(11):
        for b in range(6):
            for c in range(3):
                for d in range(2):
                    val = (2 ** a) * (3 ** b) * (5 ** c) * (7 ** d)
                    if val > 12:
                        divisors.add(val)
    
    # Sort divisors in descending order for the search
    # Descending order helps in pruning: if we pick a large factor, remaining drops fast.
    # Also, in the feasibility check, if candidate > remaining, we skip it.
    # If sorted descending, once candidate <= remaining, all subsequent are also <= remaining?
    # No, if sorted descending, candidate[i] > candidate[i+1].
    # If candidate[i] > remaining, candidate[i+1] might be <= remaining.
    # So we can't break, we must continue skipping until we find one <= remaining.
    
    sorted_divs_desc = sorted(divisors, reverse=True)
    
    # 2. Feasibility Check using DP/DFS with Memoization
    # We binary search on the maximum allowed factor M.
    
    def is_feasible(M):
        # Candidates are divisors <= M
        # Since sorted_divs_desc is sorted descending, we can find the cutoff index
        # But it's easier to just filter or use the whole list and check condition
        
        # To optimize, let's create the list of candidates for this M
        # candidates = [d for d in sorted_divs_desc if d <= M]
        # This takes O(N). N=385. This is fine.
        
        candidates = [d for d in sorted_divs_desc if d <= M]
        if not candidates:
            return False
        
        n = len(candidates)
        
        @lru_cache(maxsize=None)
        def dfs(remaining, idx, count):
            if time.time() > deadline:
                return False
            
            if remaining == 1:
                return True
            
            if count >= 12:
                return False
            
            if idx >= n:
                return False
            
            # If the current candidate is greater than remaining, we cannot use it.
            # We must skip it.
            if candidates[idx] > remaining:
                return dfs(remaining, idx + 1, count)
            
            # Option 1: Skip current candidate
            if dfs(remaining, idx + 1, count):
                return True
            
            # Option 2: Use current candidate
            if remaining % candidates[idx] == 0:
                if dfs(remaining // candidates[idx], idx + 1, count + 1):
                    return True
            
            return False
        
        result = dfs(target, 0, 0)
        dfs.cache_clear()
        return result

    # Binary Search for minimal M
    # Lower bound: The smallest possible max factor.
    # We know max factor >= 13.
    # Upper bound: target
    low = 13
    high = target
    
    while low < high:
        mid = (low + high) // 2
        if is_feasible(mid):
            high = mid
        else:
            low = mid + 1
    
    optimal_M = low
    
    # 3. Reconstruct the solution
    # We need to find the actual set of factors.
    # We can run a similar DFS that records the path.
    
    candidates = [d for d in sorted_divs_desc if d <= optimal_M]
    n = len(candidates)
    best_factors = []
    
    def find_solution(remaining, idx, count, path):
        if time.time() > deadline:
            return False
        
        if remaining == 1:
            best_factors.extend(path)
            return True
        
        if count >= 12:
            return False
        
        if idx >= n:
            return False
        
        if candidates[idx] > remaining:
            return find_solution(remaining, idx + 1, count, path)
        
        # Option 1: Skip
        if find_solution(remaining, idx + 1, count, path):
            return True
        
        # Option 2: Use
        if remaining % candidates[idx] == 0:
            path.append(candidates[idx])
            if find_solution(remaining // candidates[idx], idx + 1, count + 1, path):
                return True
            path.pop()
        
        return False
    
    if not find_solution(target, 0, 0, []):
        # Fallback: Should not happen
        return [target]
        
    # The problem asks for distinct integers. The DFS ensures distinctness by using index.
    # The problem asks for product exactly 12!. The DFS ensures this.
    # The problem asks for integers > 12. The candidates ensure this.
    # The problem asks to minimize largest factor. Binary search ensures this.
    
    return best_factors
