import time
from math import factorial
from bisect import bisect_left, bisect_right

def construct():
    start_time = time.time()
    timeout = 85.0
    
    target = factorial(12)
    
    # Get all divisors of target
    divisors = []
    for i in range(1, int(target**0.5) + 1):
        if target % i == 0:
            divisors.append(i)
            if i != target // i:
                divisors.append(target // i)
            
    # Filter valid divisors: > 12
    valid_divs = [d for d in divisors if d > 12]
    valid_divs.sort()
    
    n_divs = len(valid_divs)
    
    # Precompute prefix products for min/max pruning?
    # Not directly possible because we pick distinct indices.
    
    def can_decompose_k(M, k):
        # Find candidates <= M
        end_idx = bisect_right(valid_divs, M)
        if end_idx < k:
            return False
        candidates = valid_divs[:end_idx]
        
        # Pruning helpers
        # We need to pick k distinct elements from candidates.
        # The smallest possible product of k distinct elements from candidates[idx:]
        # is the product of the k smallest elements in candidates[idx:].
        # The largest possible product is the product of the k largest elements in candidates[idx:].
        
        # To make this fast, we can precompute suffix products?
        # But the set of available candidates changes as we pick.
        # However, since k is small (<=7), we can just compute on the fly.
        
        # Actually, a stronger bound:
        # If we are at index `idx` and need to pick `r` more factors.
        # The min product we can achieve is product of candidates[idx ... idx+r-1].
        # The max product we can achieve is product of candidates[end_idx-r ... end_idx-1].
        
        # Let's use a recursive search.
        
        res = []
        
        def backtrack(idx, remaining, count):
            if time.time() - start_time > timeout - 5:
                return False, None
            
            if count == k:
                if remaining == 1:
                    return True, res[:]
                return False, None
            
            # Pruning
            # 1. If remaining is 1, we can't pick more.
            # 2. If we need 'r' more factors.
            r = k - count
            
            # Check if enough candidates left
            if end_idx - idx < r:
                return False, None
            
            # Min possible product for remaining r factors
            # Must be distinct and from candidates[idx:]
            # The absolute minimum is picking the smallest r available: candidates[idx], ..., candidates[idx+r-1]
            min_prod = 1
            for j in range(idx, idx + r):
                min_prod *= candidates[j]
            if min_prod > remaining:
                return False, None
                
            # Max possible product for remaining r factors
            # The absolute maximum is picking the largest r available: candidates[end_idx-r], ..., candidates[end_idx-1]
            max_prod = 1
            for j in range(end_idx - r, end_idx):
                max_prod *= candidates[j]
            if max_prod < remaining:
                return False, None
                
            # If remaining is prime, it must be one of the candidates
            # (Optimization: check if remaining is in candidates[idx:])
            # But we also need to pick r-1 other factors.
            # If r==1, we just check if remaining is in candidates[idx:].
            if r == 1:
                pos = bisect_left(candidates, remaining, idx)
                if pos < end_idx and candidates[pos] == remaining:
                    res.append(remaining)
                    return True, res[:]
                return False, None
            
            # Iterate candidates
            for i in range(idx, end_idx):
                c = candidates[i]
                if c > remaining:
                    break
                if remaining % c == 0:
                    res.append(c)
                    ok, sol = backtrack(i + 1, remaining // c, count + 1)
                    if ok:
                        return True, sol
                    res.pop()
                    
            return False, None
        
        ok, sol = backtrack(0, target, 0)
        return ok

    def can_decompose(M):
        # Check if any k in [1, 7] works
        # We can try k from 1 to 7.
        # Note: larger k generally allows smaller M, but we are fixing M and checking existence.
        # If a solution exists for k=7, it exists for M.
        # We just need to return True if ANY valid k exists.
        
        # Heuristic: try k that are likely to work.
        # Actually, just loop 1..7.
        for k in range(1, 8):
            if can_decompose_k(M, k):
                return True
        return False

    # Binary search for M
    low = 13
    high = target
    found_M = None
    
    # We can narrow high.
    # For k=1, M=target.
    # For k=2, M ~ 22000.
    # For k=3, M ~ 800.
    # For k=4, M ~ 150.
    # For k=5, M ~ 90.
    # For k=6, M ~ 60.
    # For k=7, M ~ 45.
    
    # Let's just binary search.
    while low <= high:
        if time.time() - start_time > timeout - 10:
            break
        mid = (low + high) // 2
        if can_decompose(mid):
            found_M = mid
            high = mid - 1
        else:
            low = mid + 1

    if found_M is None:
        return [target]
        
    # Reconstruct solution
    # We need to find the actual factors for the smallest M.
    # We can just call can_decompose_k for the found_M and return the solution.
    # But can_decompose_k only returns bool in the check above?
    # I modified it to return (ok, sol).
    
    # Let's refine can_decompose_k to return the solution if found.
    def find_solution_k(M, k):
        end_idx = bisect_right(valid_divs, M)
        if end_idx < k:
            return None
        candidates = valid_divs[:end_idx]
        res = []
        
        def backtrack(idx, remaining, count):
            if time.time() - start_time > timeout - 2:
                return False
            
            if count == k:
                if remaining == 1:
                    return True
                return False
            
            r = k - count
            if end_idx - idx < r:
                return False
            
            # Min/Max pruning
            min_prod = 1
            for j in range(idx, idx + r):
                min_prod *= candidates[j]
            if min_prod > remaining:
                return False
                
            max_prod = 1
            for j in range(end_idx - r, end_idx):
                max_prod *= candidates[j]
            if max_prod < remaining:
                return False
                
            if r == 1:
                pos = bisect_left(candidates, remaining, idx)
                if pos < end_idx and candidates[pos] == remaining:
                    res.append(remaining)
                    return True
                return False
            
            for i in range(idx, end_idx):
                c = candidates[i]
                if c > remaining:
                    break
                if remaining % c == 0:
                    res.append(c)
                    if backtrack(i + 1, remaining // c, count + 1):
                        return True
                    res.pop()
            return False
        
        if backtrack(0, target, 0):
            return res
        return None

    best_sol = None
    # Try to find a solution with the found_M
    # It might be that found_M works for k=7, but we should return the one with min max factor.
    # Since we binary searched for the min M such that ANY k works, found_M is the global min max factor.
    # We just need to find WHICH k gives that solution.
    
    for k in range(1, 8):
        sol = find_solution_k(found_M, k)
        if sol:
            best_sol = sol
            break
            
    if best_sol is None:
        # Fallback
        return [target]
        
    return best_sol
