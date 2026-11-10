import math
import time

def construct():
    start_time = time.time()
    N = 479001600  # 12!
    
    # 1. Generate all divisors of N
    divisors = set()
    for i in range(1, int(math.isqrt(N)) + 1):
        if N % i == 0:
            divisors.add(i)
            divisors.add(N // i)
    
    # Filter divisors > 12
    valid_divs = sorted([d for d in divisors if d > 12])
    
    best_max = float('inf')
    best_factors = []
    
    # Max k is 7 because 13*...*20 > 12!
    # Let's verify: 13*14*15*16*17*18*19 = 253955520 < 479001600
    # 13*...*20 = 5079110400 > 479001600
    # So k can be at most 7.
    
    # We try k from 7 down to 1.
    # For each k, we want to find a set of k distinct divisors from valid_divs
    # such that their product is N, and the maximum element is minimized.
    
    # To speed up, we can precompute prefix products or just rely on the small size.
    # len(valid_divs) is around 700.
    # DFS depth 7.
    # Branching factor is reduced by product constraint and best_max bound.
    
    def dfs(start_idx, current_prod, count, path, target_k):
        nonlocal best_max, best_factors
        
        if time.time() - start_time > 85:
            return False
        
        if count == target_k:
            if current_prod == N:
                max_val = path[-1] # Since sorted, last is max
                if max_val < best_max:
                    best_max = max_val
                    best_factors = list(path)
                    return True
            return False
        
        remaining = target_k - count
        # The next divisor must be at least valid_divs[start_idx]
        # And we need 'remaining' more divisors.
        # The smallest possible product we can form with 'remaining' divisors
        # starting from index start_idx is the product of the 'remaining' smallest
        # available divisors.
        # However, they must divide N/current_prod.
        
        # Pruning 1: If current_prod > N, return
        if current_prod > N:
            return False
            
        # Pruning 2: Lower bound on remaining product
        # The smallest 'remaining' divisors from valid_divs[start_idx:]
        # Note: We must ensure they are distinct and > last chosen.
        # Since we pick indices in increasing order, they are distinct and > last chosen.
        min_prod_rem = 1
        for i in range(remaining):
            idx = start_idx + i
            if idx >= len(valid_divs):
                return False
            min_prod_rem *= valid_divs[idx]
        
        if current_prod * min_prod_rem > N:
            return False
            
        # Pruning 3: Upper bound on candidate
        # If we pick d, the max will be at least d.
        # If d >= best_max, we can't improve.
        # Also, d must divide N/current_prod.
        
        # Iterate through candidates
        # We can limit the range of start_idx to len(valid_divs) - remaining
        for i in range(start_idx, len(valid_divs) - remaining + 1):
            d = valid_divs[i]
            
            # If d is already >= best_max, no point continuing
            # because the max will be at least d
            if d >= best_max:
                break
                
            # d must divide the remaining quotient
            rem_quotient = N // current_prod
            if rem_quotient % d != 0:
                continue
                
            path.append(d)
            if dfs(i + 1, current_prod * d, count + 1, path, target_k):
                # If we found a better solution, we can stop?
                # Not necessarily, there might be a better one for this k?
                # No, we are searching for min max.
                # If we found one with max d, and we are iterating d in increasing order?
                # No, we are iterating indices.
                # If we found a solution, best_max is updated.
                # The loop `if d >= best_max: break` will prune subsequent candidates
                # in the *current* call stack if they are larger.
                # But we should probably continue to see if a smaller max is possible?
                # Wait, if we found a solution with max `d_last`, and we are at index `i`
                # where `d` is the first element.
                # The max of the solution found is `path[-1]`.
                # If we return True, we bubble up.
                # But we might find a solution with an even smaller max by choosing
                # a different first element?
                # Yes. So we shouldn't return True immediately and stop the whole search
                # for this k. We should just update best_max and continue.
                # However, the `break` condition `d >= best_max` handles pruning.
                pass
            
            path.pop()
            
        return False

    # Try k from 7 down to 1
    # We can stop early if best_max is very small, but 13 is the min possible factor.
    # So if best_max == 13, we can't do better (unless k=1 and factor=13, but 13!=12!).
    # Actually, min factor is 13. Min max is at least 13.
    
    for k in range(7, 0, -1):
        # Optional: quick check if min product of k smallest divisors > N
        min_prod = 1
        for i in range(k):
            if i >= len(valid_divs):
                break
            min_prod *= valid_divs[i]
        if min_prod > N:
            continue
            
        dfs(0, 1, 0, [], k)
        
        # If we found a solution with max 13, it's optimal?
        # Not necessarily, but 13 is the smallest possible value for any factor.
        # So max cannot be less than 13.
        if best_max == 13:
            break

    if not best_factors:
        # Fallback, though shouldn't happen
        return [N]
        
    return best_factors
