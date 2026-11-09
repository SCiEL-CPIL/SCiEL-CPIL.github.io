import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    target = math.factorial(12)  # 479001600
    
    # Get all divisors of target
    divisors = set()
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            divisors.add(i)
            divisors.add(target // i)
    
    # Filter: must be > 12, sort ascending
    candidates = sorted([d for d in divisors if d > 12])
    n_cand = len(candidates)
    
    # Precompute for each candidate index, the list of (idx, value) where value divides target
    # and value is at position >= some index. Actually, let's precompute divisors of each remainder.
    
    # For fast lookup: for a given remainder r, which candidate indices have candidates[i] | r?
    # We'll build this lazily with caching.
    
    def _is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    # Memoization cache: (rem, start_idx) -> True/False for a given max_val
    # Since max_val changes, we need to be careful. We'll use a dict keyed by (max_val, rem, start_idx)
    # But that's too large. Instead, for each M we do a fresh DFS with memo.
    
    def can_decompose(rem, start_idx, max_val, path, memo):
        """Try to decompose 'rem' into distinct factors from candidates[start_idx:] all <= max_val."""
        if time.time() > deadline:
            return None
        
        if rem == 1:
            return path[:]
        
        if len(path) >= 12:
            return None
        
        if rem < 13:
            return None
        
        # Check memo
        key = (rem, start_idx)
        if key in memo:
            if memo[key] is None:
                return None
            # memo[key] is True means feasible, but we need the actual path
            # So memo only stores infeasibility
            return None
        
        # If rem itself is > max_val and is prime, we can't use it directly
        # and can't factor it into numbers all <= max_val (since prime)
        if rem > max_val:
            if _is_prime(rem):
                memo[key] = None
                return None
            # rem > max_val but composite: we need a factor f of rem with 12 < f <= max_val
            # such that rem/f can also be decomposed. This is handled by the loop below.
        
        # Try each candidate from start_idx
        for i in range(start_idx, n_cand):
            f = candidates[i]
            if f > max_val:
                break
            if f > rem:
                break
            if rem % f != 0:
                continue
            
            new_rem = rem // f
            new_path = path + [f]
            result = can_decompose(new_rem, i + 1, max_val, new_path, memo)
            if result is not None:
                return result
        
        memo[key] = None
        return None
    
    # Binary search on M: find smallest M such that decomposition exists
    lo = 0
    hi = n_cand - 1
    best_solution = None
    best_M = None
    
    while lo <= hi:
        mid = (lo + hi) // 2
        M = candidates[mid]
        
        if time.time() > deadline:
            break
        
        memo = {}
        sol = can_decompose(target, 0, M, [], memo)
        
        if sol is not None:
            best_solution = sol
            best_M = M
            hi = mid - 1  # Try smaller
        else:
            lo = mid + 1  # Need larger
    
    if best_solution is None:
        # Fallback: try the largest candidate as single factor
        return [target]
    
    return best_solution
