import math
import time
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    target = math.factorial(12)
    
    # Get all divisors of target
    divisors = set()
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            divisors.add(i)
            divisors.add(target // i)
    
    # Filter: must be > 12
    candidates = sorted([d for d in divisors if d > 12])
    n_cands = len(candidates)
    
    # Max possible factors is 7
    MAX_FACTORS = 7
    
    # We will binary search M
    # But we can be smarter.
    # The minimal possible max factor is at least ceil((target)^(1/k)) for some k.
    # Let's just binary search M over the candidates.
    
    best_solution = None
    
    # To speed up can_decompose, we precompute for each index i, the list of candidates from i onwards?
    # No, just use the global candidates list and index.
    
    # We'll use a helper that takes M and returns a solution or None.
    # Since M changes, we can't share cache across M values easily unless we include M in key.
    # But binary search calls can_decompose with different M.
    
    def find_solution_for_M(M):
        # M is the maximum allowed factor.
        # We need to find distinct factors from candidates, all <= M, product target.
        
        # Memoization key: (rem, start_idx)
        # Value: list of factors or None
        # But list is not hashable. We can return bool and store path separately?
        # Or just use a dict for memo.
        
        memo = {}
        
        def dfs(rem, start_idx, path):
            if rem == 1:
                return path[:]
            
            if len(path) >= MAX_FACTORS:
                return None
            
            if start_idx >= n_cands:
                return None
            
            # Pruning:
            # 1. If rem < 13, fail (since next factor must be > 12)
            if rem < 13:
                return None
            
            # 2. If the smallest available factor is > rem, fail
            if candidates[start_idx] > rem:
                return None
                
            # 3. If rem is prime, it must be the last factor.
            # Check if rem is in candidates and > start_idx and <= M.
            # Actually, if rem is prime, it has only 1 and rem as divisors.
            # Since factors must be > 12, rem must be one of the factors.
            # So we just check if rem is a valid candidate.
            # But we also need to ensure it's distinct and not already in path.
            # Since we pick in increasing order, if rem > candidates[start_idx], it's distinct.
            # Wait, if rem is prime, it must be equal to one of the factors.
            # If we pick rem as the next factor, the product of remaining must be 1.
            # So if rem is prime, we can just return [rem] if rem <= M and rem > 12 and rem is in candidates.
            # But we need to check if rem is actually in the candidates list (i.e., divides target).
            # Since rem divides target (by construction), if rem > 12, it is in candidates.
            # So if is_prime(rem):
            #   if rem <= M: return path + [rem]
            #   else: return None
            
            # Let's implement is_prime check for speed?
            # rem is at most 12!.
            
            # 4. General pruning:
            # The minimum product we can form with `k` more factors starting from `start_idx`
            # is the product of the k smallest available candidates.
            # The maximum product is the product of the k largest available candidates <= M.
            # If rem < min_product or rem > max_product, fail.
            
            # Let's stick to simple pruning first.
            
            key = (rem, start_idx)
            if key in memo:
                return memo[key]
            
            result = None
            
            # Iterate candidates
            for i in range(start_idx, n_cands):
                f = candidates[i]
                if f > M:
                    break
                if f > rem:
                    break
                if rem % f != 0:
                    continue
                
                # Optimization: If we pick f, the remaining product is rem/f.
                # The next factors must be > f.
                new_rem = rem // f
                new_path = path + [f]
                
                res = dfs(new_rem, i + 1, new_path)
                if res is not None:
                    result = res
                    memo[key] = result
                    return result
            
            memo[key] = None
            return None
        
        return dfs(target, 0, [])
    
    # Binary search for M
    # Lower bound: smallest possible max factor.
    # Upper bound: target (single factor)
    
    lo = 0
    hi = n_cands - 1
    
    # We want the smallest M such that find_solution_for_M(M) is not None.
    # But find_solution_for_M is expensive.
    # We can try to estimate a good lower bound.
    # For k=7, min max factor is approx 36.
    # Let's start lo from the index of the smallest candidate > 36?
    # Or just binary search.
    
    # To avoid running out of time, we limit the number of binary search steps?
    # Or just rely on the deadline.
    
    while lo <= hi:
        if time.time() > deadline:
            break
            
        mid = (lo + hi) // 2
        M = candidates[mid]
        
        sol = find_solution_for_M(M)
        
        if sol is not None:
            best_solution = sol
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_solution is None:
        # Fallback: try to find ANY solution, maybe with larger M
        # The binary search should have found one if it exists.
        # If not, return [target]
        return [target]
        
    return best_solution
