import math
import time
import bisect

def construct():
    start_time = time.time()
    N = 479001600  # 12!
    
    # Precompute divisors of N for fast lookup
    divisors = set()
    for i in range(1, int(N**0.5) + 1):
        if N % i == 0:
            divisors.add(i)
            divisors.add(N // i)
    divisors_list = sorted(divisors)
    
    best_factors = [N]
    best_max_val = N
    
    def can_complete(target_rem, remaining, last_factor):
        """Check if target_rem can be factored into 'remaining' distinct integers > last_factor."""
        if remaining == 1:
            return target_rem > last_factor and target_rem > 12
        if remaining == 2:
            # Need two distinct integers a, b > last_factor with a*b = target_rem
            # a < b, so a < sqrt(target_rem)
            a_max = int(math.isqrt(target_rem))
            for a in range(max(last_factor + 1, 13), a_max + 1):
                if target_rem % a == 0:
                    b = target_rem // a
                    if b > a and b > last_factor and b > 12:
                        return True
            return False
        # For remaining >= 3, use min product check
        min_prod = 1
        for i in range(1, remaining + 1):
            min_prod *= (last_factor + i)
        if target_rem < min_prod:
            return False
        return True
    
    def dfs(target_k, current_prod, last_factor, depth, path):
        nonlocal best_max_val, best_factors
        
        # Time check: keep a safety margin
        if time.time() - start_time > 85:
            return
        
        if depth == target_k:
            if current_prod == N:
                max_f = max(path)
                if max_f < best_max_val:
                    best_max_val = max_f
                    best_factors = list(path)
            return
        
        remaining = target_k - depth
        
        if N % current_prod != 0:
            return
        target_rem = N // current_prod
        
        # Feasibility check
        if not can_complete(target_rem, remaining, last_factor):
            return
        
        # If remaining == 1, the next factor must be target_rem
        if remaining == 1:
            x = target_rem
            if x > last_factor and x > 12:
                path.append(x)
                dfs(target_k, current_prod * x, x, depth + 1, path)
                path.pop()
            return
        
        lower_bound = max(last_factor + 1, 13)
        
        # Estimate upper bound for x
        # x * (x+1) * ... * (x+remaining-1) <= target_rem
        # x^remaining <= target_rem => x <= target_rem^(1/remaining)
        # Use a tighter bound
        x_upper = int(target_rem ** (1.0 / remaining)) + 5
        
        # Find divisors of N in [lower_bound, x_upper] that also divide target_rem
        left_idx = bisect.bisect_left(divisors_list, lower_bound)
        right_idx = bisect.bisect_right(divisors_list, x_upper)
        
        # Collect valid candidates and sort by proximity to ideal value
        ideal = target_rem ** (1.0 / remaining)
        candidates = []
        for idx in range(left_idx, right_idx):
            x = divisors_list[idx]
            if x > last_factor and x > 12 and target_rem % x == 0:
                candidates.append(x)
        
        # Sort by closeness to ideal value to find balanced solutions first
        candidates.sort(key=lambda x: abs(math.log(x) - math.log(ideal)))
        
        for x in candidates:
            if time.time() - start_time > 85:
                return
            path.append(x)
            dfs(target_k, current_prod * x, x, depth + 1, path)
            path.pop()
            # If we found a good solution, we can prune
            if best_max_val <= x_upper:
                # Any solution with a factor >= x_upper won't improve
                break

    # Try k=7 first (most factors → smallest individual values)
    for k in range(7, 1, -1):
        dfs(k, 1, 12, 0, [])
        if best_max_val <= 100:
            break
        
    return best_factors

# EVOLVE-BLOCK-START
# The above code is the logic. I will wrap it in the required format.
# EVOLVE-BLOCK-END
