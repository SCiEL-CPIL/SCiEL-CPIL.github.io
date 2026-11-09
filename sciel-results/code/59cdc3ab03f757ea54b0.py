import time
from math import factorial
from functools import lru_cache

_START = time.time()
_DEADLINE = _START + 85.0  # safety margin before 90s kill

def _check_deadline():
    return time.time() < _DEADLINE

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 85.0
    
    target = factorial(12)  # 479001600 = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    
    # Generate all divisors of target
    # 2^a * 3^b * 5^c * 7^d * 11^e where a=0..10, b=0..5, c=0..2, d=0..1, e=0..1
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
    
    # Best solution found
    best_max = [target]  # initial: single factor
    best_factors = [[target]]
    
    # DFS: find distinct divisors whose product = target, minimize max
    # Pruning: 
    # 1. current_max >= best_max[0] -> prune
    # 2. product so far * (min possible remaining) > target -> prune  
    # 3. product so far doesn't divide target -> prune (shouldn't happen if we pick divisors)
    # 4. len(factors) >= 12 -> prune
    
    def dfs(idx, current_product, current_max, factors, used):
        if not _check_deadline():
            return
        if current_product == target:
            if current_max < best_max[0]:
                best_max[0] = current_max
                best_factors[0] = list(factors)
            return
        if current_product > target:
            return
        if len(factors) >= 12:
            return
        # Prune: if current_max already >= best, no point continuing
        if current_max >= best_max[0]:
            return
        
        # Try each divisor from idx onwards
        for i in range(idx, n_divs):
            if not _check_deadline():
                return
            d = valid_divs[i]
            if d in used:
                continue
            if current_product * d > target:
                break  # since sorted, all further will be larger
            # Check divisibility
            if target % current_product != 0:
                continue
            remaining = target // current_product
            if d > remaining:
                break
            if remaining % d != 0:
                continue
            
            new_max = max(current_max, d)
            if new_max >= best_max[0]:
                continue
            
            factors.append(d)
            used.add(d)
            dfs(i + 1, current_product * d, new_max, factors, used)
            factors.pop()
            used.remove(d)
    
    # Start DFS
    dfs(0, 1, 0, [], set())
    
    return best_factors[0]
