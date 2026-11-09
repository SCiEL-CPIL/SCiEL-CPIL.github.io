import math
import time

def construct():
    target = math.factorial(12)
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Precompute all divisors of 12!
    divisors = []
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            divisors.append(i)
            if i != target // i:
                divisors.append(target // i)
    
    # Filter divisors > 12
    candidates = [d for d in divisors if d > 12]
    candidates.sort()
    
    # Binary search for the minimum max factor M
    low = 13
    high = target
    best_factors = None
    
    def can_form(M):
        valid = [d for d in candidates if d <= M]
        if not valid:
            return False, None
        
        import bisect
        
        # We use a set to track visited states to avoid redundant work in this specific M check?
        # Or just rely on DFS structure.
        # Let's add a simple cache for (rem_target, min_factor, depth)
        # But min_factor is continuous? No, it's always a value from valid or 12.
        # So we can use a dict.
        
        cache = {}
        
        def dfs(rem_target, min_factor, depth):
            if rem_target == 1:
                return True, []
            
            if depth >= 12:
                return False, None
            
            key = (rem_target, min_factor, depth)
            if key in cache:
                return cache[key]
            
            start_idx = bisect.bisect_right(valid, min_factor)
            
            if start_idx == len(valid):
                cache[key] = (False, None)
                return False, None
            
            # Pruning: Max product
            max_factors_allowed = 12 - depth
            available_count = len(valid) - start_idx
            
            # If we can't even pick max_factors_allowed, we pick all available
            # Product of largest min(max_factors_allowed, available_count) elements
            if available_count <= max_factors_allowed:
                max_prod = 1
                for f in valid[start_idx:]:
                    max_prod *= f
            else:
                # Take the largest max_factors_allowed from the end of valid
                # valid is sorted, so valid[-1] is largest.
                # We need elements > min_factor.
                # So we look at valid[start_idx:]
                # The largest ones are at the end of valid.
                # But we must ensure they are > min_factor.
                # Since valid is sorted, the largest elements are definitely > min_factor 
                # unless min_factor is very large.
                # bisect_right ensures valid[start_idx:] are all > min_factor.
                # So the largest max_factors_allowed elements in valid[start_idx:] are 
                # simply the last max_factors_allowed elements of valid, 
                # PROVIDED that valid[start_idx:] has at least max_factors_allowed elements.
                
                # Wait, if start_idx is close to len(valid), we might not have enough.
                # We handled available_count <= max_factors_allowed above.
                
                max_prod = 1
                # Indices: len(valid) - max_factors_allowed to len(valid)
                # But we must start from start_idx.
                # So indices: max(start_idx, len(valid) - max_factors_allowed) to len(valid)
                start_max = max(start_idx, len(valid) - max_factors_allowed)
                for f in valid[start_max:]:
                    max_prod *= f
            
            if max_prod < rem_target:
                cache[key] = (False, None)
                return False, None
            
            # Pruning: Min product
            # Smallest available is valid[start_idx]
            if valid[start_idx] > rem_target:
                cache[key] = (False, None)
                return False, None
            
            for i in range(start_idx, len(valid)):
                f = valid[i]
                if f > rem_target:
                    break
                if rem_target % f != 0:
                    continue
                
                ok, sub_factors = dfs(rem_target // f, f, depth + 1)
                if ok:
                    cache[key] = (True, [f] + sub_factors)
                    return True, [f] + sub_factors
            
            cache[key] = (False, None)
            return False, None
        
        ok, factors = dfs(target, 12, 0)
        return ok, factors

    while low <= high:
        if time.time() > deadline:
            break
        mid = (low + high) // 2
        ok, factors = can_form(mid)
        if ok:
            best_factors = factors
            high = mid - 1
        else:
            low = mid + 1
    
    if best_factors:
        return best_factors
    else:
        return [target]
