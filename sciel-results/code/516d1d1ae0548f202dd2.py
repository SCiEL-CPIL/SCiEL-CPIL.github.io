import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = math.factorial(12)
    
    # Generate all divisors of N
    divisors = []
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    exps = [10, 5, 2, 1, 1]
    
    def gen_divs(idx, current_val):
        if idx == len(primes):
            if current_val > 12:
                divisors.append(current_val)
            return
        for e in range(exps[idx] + 1):
            gen_divs(idx + 1, current_val * (primes[idx] ** e))
            
    gen_divs(0, 1)
    divisors.sort()
    
    # Binary search on the maximum allowed factor
    # The max factor must be one of the divisors.
    # We want the smallest max factor M such that a valid decomposition exists using factors <= M.
    
    def can_form_with_max(max_val):
        # Filter divisors <= max_val
        valid_divs = [d for d in divisors if d <= max_val]
        if not valid_divs:
            return False, None
        
        # DFS to find if a subset of valid_divs multiplies to N
        # State: index, current_product, count, path
        # We want to find ANY valid path.
        
        n = len(valid_divs)
        
        # Memoization might help? State is (index, current_product, count).
        # current_product can be large, but it's a divisor of N.
        # There are only 792 divisors. So current_product is one of them.
        # Memo key: (index, current_product, count)
        # index up to ~792, product up to 792 values, count up to 12.
        # Total states ~ 792 * 792 * 12 ~ 7.5 million. This is feasible.
        
        memo = {}
        
        def dfs(idx, curr_prod, count, path):
            if time.time() > deadline:
                return False, None
            
            if curr_prod == N:
                return True, path[:]
            
            if count == 12:
                return False, None
            
            if idx == n:
                return False, None
            
            # Pruning: if curr_prod > N, impossible
            if curr_prod > N:
                return False, None
                
            # Pruning: if N % curr_prod != 0, impossible
            if N % curr_prod != 0:
                return False, None
                
            rem = N // curr_prod
            
            # If rem is 1, handled above.
            
            # Check if rem can be formed by divisors from idx onwards.
            # If the smallest divisor valid_divs[idx] > rem, we can't form rem (since all others are larger).
            # Wait, we need product of subset = rem.
            # If min(valid_divs[idx:]) > rem, then any product of >=1 elements is > rem.
            # So if rem != 1, and min > rem, return False.
            if valid_divs[idx] > rem:
                return False, None
            
            key = (idx, curr_prod, count)
            if key in memo:
                return memo[key]
            
            # Option 1: Skip valid_divs[idx]
            res, p = dfs(idx + 1, curr_prod, count, path)
            if res:
                memo[key] = (res, p)
                return res, p
            
            # Option 2: Take valid_divs[idx]
            d = valid_divs[idx]
            if rem % d == 0:
                path.append(d)
                res, p = dfs(idx + 1, curr_prod * d, count + 1, path)
                path.pop()
                if res:
                    memo[key] = (res, p)
                    return res, p
            
            memo[key] = (False, None)
            return False, None
        
        return dfs(0, 1, 0, [])

    # Binary search
    lo = 0
    hi = len(divisors) - 1
    best_solution = None
    
    # We know a solution exists (e.g., {N} is not valid if N<=12? No N is huge. {13, ...}?)
    # Actually, is a solution always possible?
    # 12! = 479001600.
    # We can always use {N} if N > 12. But we want to minimize max.
    # We want to split into smaller factors.
    
    while lo <= hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        max_val = divisors[mid]
        
        res, path = can_form_with_max(max_val)
        
        if res:
            best_solution = path
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_solution:
        return best_solution
    
    # Fallback: return a valid solution found by simple search or just [N] if valid?
    # [N] is valid if N > 12 and distinct (count 1).
    # But we want to minimize max.
    # If binary search fails, maybe return the first found in a linear scan?
    # Or just return [N] as a valid (but suboptimal) solution to avoid 0 score.
    # However, the task says "Minimize the LARGEST factor".
    # The evaluator scores based on the value returned.
    # Returning [N] gives score N.
    # Returning a better split gives lower score.
    
    # If best_solution is None, it means no solution found within time.
    # We should return the best candidate we found during the search?
    # The binary search only returns the final result.
    # We can modify can_form_with_max to keep track of the best path found for ANY max?
    # No, the binary search structure is: if mid works, try smaller.
    # If mid doesn't work, try larger.
    # The first "works" found in a linear scan from bottom would be the optimal.
    # Binary search finds the boundary.
    
    # To be safe, let's just do a linear scan from smallest divisor upwards?
    # There are ~790 divisors.
    # Each check is a DFS.
    # If the optimal max is small, the DFS is fast.
    # If the optimal max is large, the DFS might be slower, but we stop at the first success.
    # Linear scan:
    # for i in range(len(divisors)):
    #   if can_form_with_max(divisors[i]):
    #      return path
    
    # This guarantees we find the smallest max.
    # Risk: if the smallest max is large, we do many DFS calls.
    # But we can stop as soon as we find one.
    # The "can_form" for small max values will fail quickly (pruning: min_div > rem).
    # So linear scan is probably faster than binary search in practice for this specific "minimize max" objective, because the "fail" cases are cheap.
    
    # Let's revert to linear scan for simplicity and guaranteed optimality (first success is best).
    
    for i in range(len(divisors)):
        if time.time() > deadline:
            break
        max_val = divisors[i]
        res, path = can_form_with_max(max_val)
        if res:
            return path
            
    return [N]
