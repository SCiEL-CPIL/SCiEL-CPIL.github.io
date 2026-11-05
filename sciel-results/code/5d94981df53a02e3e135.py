import time
from math import factorial
from itertools import product

_START_TIME = time.time()
_TIMEOUT = 85

def _time_left():
    return _TIMEOUT - (time.time() - _START_TIME)

def construct():
    target = factorial(12)  # 479001600
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # Prime exponents: [10, 5, 2, 1, 1]
    primes = [2, 3, 5, 7, 11]
    target_exp = [10, 5, 2, 1, 1]
    
    # Generate all divisors > 12 as exponent vectors
    divisors = []
    for e0 in range(11):
        for e1 in range(6):
            for e2 in range(3):
                for e3 in range(2):
                    for e4 in range(2):
                        val = (2**e0) * (3**e1) * (5**e2) * (7**e3) * (11**e4)
                        if val > 12:
                            divisors.append((val, (e0, e1, e2, e3, e4)))
    
    divisors.sort()
    
    best_result = None
    best_max = target + 1
    
    # Try increasing max bounds
    for M in range(24, 500):
        if _time_left() < 5:
            break
        
        valid = [(val, exp) for val, exp in divisors if val <= M]
        if len(valid) < 1:
            continue
        
        found = [None]
        
        def dfs(rem_exp, min_idx, path, used_vals):
            if _time_left() < 0.5:
                return
            if all(x == 0 for x in rem_exp):
                if path:
                    found[0] = list(path)
                return
            if min_idx >= len(valid):
                return
            
            # Pruning: remaining product must be achievable
            rem_product = 1
            for p, e in zip(primes, rem_exp):
                rem_product *= p ** e
            if rem_product == 1:
                if path:
                    found[0] = list(path)
                return
            if rem_product == 0:
                return
            
            # Lower bound on number of more factors needed
            # Each factor > 12, so at most rem_product // 13 more factors can be used
            # But we need at least 1 more factor
            if len(path) >= 12:
                return
            
            for i in range(min_idx, len(valid)):
                val, exp = valid[i]
                if val in used_vals:
                    continue
                # Check if this divisor can be applied
                can_apply = True
                new_rem = []
                for j in range(5):
                    if exp[j] > rem_exp[j]:
                        can_apply = False
                        break
                    new_rem.append(rem_exp[j] - exp[j])
                if not can_apply:
                    continue
                # Check if remaining product is >= 1 (always true if can_apply)
                # Additional pruning: remaining product must be >= 13 (next factor > 12)
                # unless this is the last factor
                rem_prod = 1
                for p, e in zip(primes, new_rem):
                    rem_prod *= p ** e
                if rem_prod > 1 and rem_prod < 13:
                    continue
                
                path.append(val)
                used_vals.add(val)
                dfs(tuple(new_rem), i + 1, path, used_vals)
                if found[0] is not None:
                    path.pop()
                    used_vals.discard(val)
                    return
                path.pop()
                used_vals.discard(val)
        
        dfs(tuple(target_exp), 0, [], set())
        if found[0] is not None:
            return found[0]
    
    # Fallback: return a known valid solution
    # 12! = 479001600 = 33 * 14515200
    return [33, 14515200]
