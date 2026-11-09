import time
from math import factorial

_START = time.time()
_DEADLINE = _START + 85  # safety margin before 90s kill

def construct():
    target = factorial(12)  # 479001600 = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    
    # Generate all divisors of target
    divisors = set()
    for a in range(11):  # 2^a, a=0..10
        for b in range(6):  # 3^b, b=0..5
            for c in range(3):  # 5^c, c=0..2
                for d in range(2):  # 7^d, d=0..1
                    for e in range(2):  # 11^e, e=0..1
                        val = (2**a) * (3**b) * (5**c) * (7**d) * (11**e)
                        divisors.add(val)
    
    # Filter: must be > 12
    valid_divs = sorted([d for d in divisors if d > 12])
    
    best_max = [target]  # initial: single factor
    best_factors = [target]
    
    def dfs(remaining, start_idx, current_factors, current_max):
        if time.time() > _DEADLINE:
            return
        
        if remaining == 1:
            if len(current_factors) >= 1:
                if current_max < best_max[0]:
                    best_max[0] = current_max
                    best_factors[:] = current_factors[:]
            return
        
        if len(current_factors) >= 12:
            return
        
        # Prune: if current_max is already >= best, no point continuing
        if current_max >= best_max[0]:
            return
        
        # Try each divisor from start_idx onward
        for i in range(start_idx, len(valid_divs)):
            d = valid_divs[i]
            if d > remaining:
                break
            if remaining % d != 0:
                continue
            
            new_max = max(current_max, d)
            if new_max >= best_max[0]:
                # Since valid_divs is sorted, all subsequent d will be >= this, so new_max won't decrease
                # But wait, we could still have a better solution with fewer factors... 
                # Actually if new_max >= best_max[0], and we're adding more factors, the max can only stay or increase
                break
            
            current_factors.append(d)
            dfs(remaining // d, i + 1, current_factors, new_max)
            current_factors.pop()
    
    dfs(target, 0, [], 0)
    
    return best_factors
