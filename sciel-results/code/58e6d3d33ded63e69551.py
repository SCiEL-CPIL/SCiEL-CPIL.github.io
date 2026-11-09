import time
import bisect
from math import factorial

_START = time.time()
_DEADLINE = _START + 85

def construct():
    target = factorial(12)  # 479001600
    
    # Generate all divisors of target > 12
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    divisors = set()
    for a in range(11):
        for b in range(6):
            for c in range(3):
                for d in range(2):
                    for e in range(2):
                        val = (2**a) * (3**b) * (5**c) * (7**d) * (11**e)
                        if val > 12:
                            divisors.add(val)
    
    valid_divs = sorted(divisors)
    valid_divs_set = set(valid_divs)
    
    best_max = [target]
    best_factors = [[target]]
    
    def time_up():
        return time.time() > _DEADLINE
    
    # Precompute: for each count k, what's the minimum product of k distinct divisors > 12?
    # This helps prune: if min_product(count+1) > remaining, we can't complete
    min_prod = [1] * 13
    for i in range(1, 13):
        min_prod[i] = min_prod[i-1] * valid_divs[i-1] if i-1 < len(valid_divs) else float('inf')
    
    def dfs(last_val, remaining, current_max, count, current_factors):
        if time_up():
            return
        
        if remaining == 1:
            if count >= 1:
                if current_max < best_max[0]:
                    best_max[0] = current_max
                    best_factors[0] = list(current_factors)
            return
        
        if count >= 12:
            return
            
        if current_max >= best_max[0]:
            return
        
        # Prune: even if we use the smallest possible remaining factors, can we reach 'remaining'?
        # We need (12 - count) more factors, all > last_val
        slots_left = 12 - count
        if slots_left > 0:
            # Minimum product of slots_left distinct factors all > last_val
            min_possible = 1
            idx = bisect.bisect_right(valid_divs, last_val)
            for j in range(slots_left):
                if idx + j >= len(valid_divs):
                    min_possible = float('inf')
                    break
                min_possible *= valid_divs[idx + j]
            if min_possible > remaining:
                return
        
        min_d = max(last_val + 1, 13)
        max_d = min(remaining, best_max[0] - 1)
        
        if min_d > max_d:
            return

        start_idx = bisect.bisect_left(valid_divs, min_d)
        end_idx = bisect.bisect_right(valid_divs, max_d)
        
        for i in range(start_idx, end_idx):
            d = valid_divs[i]
            if remaining % d != 0:
                continue
                
            new_remaining = remaining // d
            new_max = max(current_max, d)
            
            if new_max >= best_max[0]:
                continue
                
            # Check if new_remaining can be factored into distinct factors > d
            if new_remaining > 1:
                # Need at least one more factor > d
                if new_remaining <= d:
                    continue
                # Check if new_remaining has any divisor > d and <= best_max[0]-1
                # Quick check: if new_remaining is prime or too small
                if new_remaining <= 13:
                    continue
                
            current_factors.append(d)
            
            if new_remaining == 1:
                dfs(d, 1, new_max, count + 1, current_factors)
            else:
                dfs(d, new_remaining, new_max, count + 1, current_factors)
            
            current_factors.pop()

    # Try to find a good initial solution quickly by greedy approach
    # Start with the smallest factors and see if we can complete
    greedy_factors = []
    remaining = target
    last = 12
    for d in valid_divs:
        if d > 12 and d > last and remaining % d == 0:
            greedy_factors.append(d)
            remaining //= d
            last = d
            if remaining == 1:
                break
            if len(greedy_factors) >= 12:
                break
    
    if remaining == 1 and len(greedy_factors) >= 1:
        greedy_max = max(greedy_factors)
        if greedy_max < best_max[0]:
            best_max[0] = greedy_max
            best_factors[0] = list(greedy_factors)
    
    dfs(0, target, 0, 0, [])
    
    return best_factors[0]
