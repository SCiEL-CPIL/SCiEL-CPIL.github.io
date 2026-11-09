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
    
    best_max = [target]
    best_factors = [[target]]
    
    def time_up():
        return time.time() > _DEADLINE

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
                
            current_factors.append(d)
            
            if new_remaining == 1:
                dfs(d, 1, new_max, count + 1, current_factors)
            else:
                if new_remaining >= d + 1:
                    dfs(d, new_remaining, new_max, count + 1, current_factors)
            
            current_factors.pop()

    dfs(0, target, 0, 0, [])
    
    return best_factors[0]
