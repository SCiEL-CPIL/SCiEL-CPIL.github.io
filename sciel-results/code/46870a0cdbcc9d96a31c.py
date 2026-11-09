import math
import time
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 45  # Safety margin for 60s limit
    
    target = math.factorial(12)  # 479001600
    
    # Get all divisors of target
    divisors = set()
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            divisors.add(i)
            divisors.add(target // i)
    
    # Filter: must be > 12
    candidates = sorted([d for d in divisors if d > 12])
    
    if not candidates:
        return [target]
        
    # If target itself is a valid candidate (which it is, 479001600 > 12)
    # The worst case is [target], max factor = target.
    
    # We want to find the minimum M such that target can be decomposed
    # into distinct factors from candidates, all <= M.
    
    # Binary search on the index of M in candidates
    # M must be one of the candidates.
    
    def can_decompose(M_idx):
        """
        Checks if target can be decomposed into distinct factors
        from candidates[0..M_idx], all <= candidates[M_idx].
        Returns the list of factors if found, else None.
        """
        M_val = candidates[M_idx]
        # We can only use candidates[0..M_idx]
        
        # Memoization for failed states: (rem, start_idx)
        fail_cache = {}
        
        def dfs(rem, start_idx):
            if time.time() > deadline:
                return None
            
            if rem == 1:
                return []
            
            if start_idx > M_idx:
                return None
                
            # Check if we have exceeded the time limit or if the state is known to fail
            if (rem, start_idx) in fail_cache:
                return None
            
            # Pruning: 
            # 1. If rem is prime and rem > M_val, it can't be formed by factors <= M_val 
            #    unless rem itself is a factor (but rem > M_val, so it's not in our set).
            #    Wait, if rem is prime and rem > M_val, we can't pick rem. 
            #    Can we form rem from smaller factors? No, because rem is prime.
            #    So if rem is prime and rem > M_val, fail.
            #    However, checking primality is expensive. 
            #    Alternative: If rem is not divisible by any candidate in [start_idx, M_idx], fail.
            
            found = False
            # Iterate through possible factors
            for i in range(start_idx, M_idx + 1):
                f = candidates[i]
                if f > rem:
                    # Since candidates are sorted, no further factor can divide rem
                    break
                
                if rem % f != 0:
                    continue
                
                # Try f
                new_rem = rem // f
                # We need to ensure distinctness, so next start is i+1
                res = dfs(new_rem, i + 1)
                if res is not None:
                    return [f] + res
                else:
                    found = True # We tried this branch, it failed
            
            if not found:
                fail_cache[(rem, start_idx)] = True
                return None
            
            return None

        return dfs(target, 0)

    # Binary search for the smallest M_idx
    lo = 0
    hi = len(candidates) - 1
    best_sol = None
    best_idx = len(candidates) - 1 # Default to largest
    
    # We know that using [target] is always a solution if target > 12.
    # target is in candidates.
    
    while lo <= hi:
        mid = (lo + hi) // 2
        sol = can_decompose(mid)
        if sol is not None:
            best_sol = sol
            best_idx = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_sol is None:
        return [target]
        
    return best_sol
