import time
import bisect
from math import factorial

_START = time.time()
_DEADLINE = _START + 55.0  # Safety margin before 60s limit

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
    n_divs = len(valid_divs)
    
    # Initial known solution to set a tight bound
    # Solution with max factor 33:
    # 33 = 3*11
    # 32 = 2^5
    # 30 = 2*3*5
    # 28 = 2^2*7
    # 27 = 3^3
    # 20 = 2^2*5
    # Total 2s: 5+1+2+2 = 10
    # Total 3s: 1+1+3 = 5
    # Total 5s: 1+1 = 2
    # Total 7s: 1
    # Total 11s: 1
    # Matches 12!
    best_factors = [33, 32, 30, 28, 27, 20]
    best_max = 33
    
    def time_up():
        return time.time() > _DEADLINE

    def dfs(last_val, remaining, current_max, count, current_factors):
        nonlocal best_max, best_factors
        
        if time_up():
            return
        
        if remaining == 1:
            if count >= 1:
                if current_max < best_max:
                    best_max = current_max
                    best_factors = list(current_factors)
            return
        
        if count >= 12:
            return
            
        # If current max is already >= best_max, we can't improve
        # Since we pick increasing factors, future max will be >= current_max
        if current_max >= best_max:
            return
            
        # The next factor must be > last_val and > 12
        min_d = max(last_val + 1, 13)
        
        # The next factor must be <= remaining (to allow division)
        # And < best_max (to allow improvement)
        max_d = min(remaining, best_max - 1)
        
        if min_d > max_d:
            return

        # Find range of divisors in [min_d, max_d]
        start_idx = bisect.bisect_left(valid_divs, min_d)
        end_idx = bisect.bisect_right(valid_divs, max_d)
        
        for i in range(start_idx, end_idx):
            if time_up():
                return
                
            d = valid_divs[i]
            
            # d must divide remaining
            if remaining % d != 0:
                continue
                
            new_remaining = remaining // d
            new_max = max(current_max, d)
            
            # If new_max >= best_max, we can break because valid_divs is sorted
            # and all subsequent d will be larger, so new_max will only increase
            if new_max >= best_max:
                break
                
            current_factors.append(d)
            
            # Recurse
            # The next factor must be > d
            dfs(d, new_remaining, new_max, count + 1, current_factors)
            
            current_factors.pop()

    dfs(0, target, 0, 0, [])
    
    return best_factors
