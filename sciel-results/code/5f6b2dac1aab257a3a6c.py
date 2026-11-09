import time
from math import factorial

_START = time.time()
_DEADLINE = _START + 85  # safety margin before 90s kill

def construct():
    target = factorial(12)  # 479001600
    
    # Generate all divisors of target > 12
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
    
    # Phase 1: Find any valid solution to get an initial upper bound
    def find_any_solution(rem, used, factors, idx):
        if time.time() > _DEADLINE:
            return None
        if rem == 1:
            return list(factors)
        if len(factors) >= 12:
            return None
        
        for i in range(idx, len(valid_divs)):
            d = valid_divs[i]
            if d > rem:
                break
            if rem % d != 0:
                continue
            if d in used:
                continue
            
            used.add(d)
            factors.append(d)
            res = find_any_solution(rem // d, used, factors, i + 1)
            if res:
                return res
            factors.pop()
            used.discard(d)
            
            if time.time() > _DEADLINE:
                return None
                
        return None

    initial_sol = find_any_solution(target, set(), [], 0)
    if initial_sol:
        best_max = max(initial_sol)
        best_factors = initial_sol
    else:
        # Fallback: single factor (though not optimal, it's valid)
        best_max = target
        best_factors = [target]
        
    # Theoretical lower bound: geometric mean for 12 factors
    # (479001600)^(1/12) ~ 23.3
    # So max factor >= 24
    lower_bound = 24
    
    if best_max == lower_bound:
        return best_factors
        
    # Phase 2: Optimize by minimizing the max factor
    # Use DFS with pruning based on current best_max
    
    def optimize_dfs(rem, used, factors, current_max, idx):
        nonlocal best_max, best_factors
        
        if time.time() > _DEADLINE:
            return
            
        if rem == 1:
            m = max(factors)
            if m < best_max:
                best_max = m
                best_factors = list(factors)
            return
            
        if len(factors) >= 12:
            return
            
        # Prune: if even using the smallest available factors we can't reach rem?
        # Or if rem is too small to be formed by remaining distinct factors?
        
        for i in range(idx, len(valid_divs)):
            d = valid_divs[i]
            
            # Prune: d must be <= rem
            if d > rem:
                break
                
            # Prune: d must be < current best_max to improve
            if d >= best_max:
                break
                
            if rem % d != 0:
                continue
                
            if d in used:
                continue
                
            new_rem = rem // d
            new_max = max(current_max, d)
            
            # Prune: if new_max >= best_max, no improvement
            if new_max >= best_max:
                continue
                
            # Additional Prune: 
            # If new_rem > 1, it must be decomposable into distinct factors > 12, < best_max
            # A necessary condition: new_rem must be divisible by at least one such factor
            # We check this in the next recursion level, but we can pre-check
            # if new_rem is not 1, does it have any valid divisor?
            if new_rem > 1:
                # Check if there exists any d' in valid_divs such that:
                # d' > 12, d' < best_max, d' not in used, d' divides new_rem
                # And d' != d (already handled by distinctness in next step? No, d is in used)
                # Actually, the next call will handle the search.
                # But if new_rem is prime and > best_max, it's impossible.
                # Or if new_rem < 13, it's impossible (since factors > 12).
                if new_rem < 13:
                    continue
                # Check divisibility by any potential next factor
                # This is a bit expensive if done naively, but valid_divs is small.
                # We can skip this for now and rely on the next level's break conditions.
                
            used.add(d)
            factors.append(d)
            optimize_dfs(new_rem, used, factors, new_max, i + 1)
            factors.pop()
            used.discard(d)
            
            if best_max == lower_bound:
                return
            if time.time() > _DEADLINE:
                return

    optimize_dfs(target, set(), [], 0, 0)
    
    return best_factors
