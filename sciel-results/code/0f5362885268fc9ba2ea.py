import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    target = math.factorial(12)
    
    # Get all divisors > 12
    divisors = []
    sqrt_n = int(math.isqrt(target))
    for i in range(1, sqrt_n + 1):
        if target % i == 0:
            if i > 12:
                divisors.append(i)
            other = target // i
            if other != i and other > 12:
                divisors.append(other)
    
    candidates = sorted(set(divisors))
    
    def can_decompose_with_max(M, time_limit):
        """Check if we can decompose target into distinct factors > 12, all <= M."""
        valid_cands = [c for c in candidates if c <= M]
        if not valid_cands:
            return None
        
        memo = {}
        
        def dfs(rem, start_idx):
            if time.time() > time_limit:
                return None
            
            if rem == 1:
                return []
            
            if start_idx >= len(valid_cands):
                return None
            
            if rem < valid_cands[start_idx]:
                return None
            
            key = (rem, start_idx)
            if key in memo:
                return memo[key]
            
            for i in range(start_idx, len(valid_cands)):
                f = valid_cands[i]
                if f > rem:
                    break
                if rem % f == 0:
                    res = dfs(rem // f, i + 1)
                    if res is not None:
                        memo[key] = [f] + res
                        return [f] + res
            
            memo[key] = None
            return None
        
        return dfs(target, 0)
    
    low = 13
    high = 1000
    
    while time.time() < deadline - 10:
        sol = can_decompose_with_max(high, deadline - 5)
        if sol is not None:
            best_sol = sol
            best_M = high
            
            while low < high and time.time() < deadline - 2:
                mid = (low + high) // 2
                sol_mid = can_decompose_with_max(mid, deadline - 2)
                if sol_mid is not None:
                    best_sol = sol_mid
                    best_M = mid
                    high = mid - 1
                else:
                    low = mid + 1
            
            return best_sol
        
        high *= 2
        if high > target:
            return [target]
    
    return [target]
