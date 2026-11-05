# EVOLVE-BLOCK-START
import time
from math import factorial

_START_TIME = time.time()
_TIMEOUT = 85  # safety margin before 90s kill

def _time_left():
    return _TIMEOUT - (time.time() - _START_TIME)

def _divisors_gt(n, min_val=13):
    """Generate all divisors of n that are >= min_val."""
    divs = set()
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            if i >= min_val:
                divs.add(i)
            j = n // i
            if j >= min_val:
                divs.add(j)
    return sorted(divs)

def _dfs(remaining, chosen, max_so_far, best):
    """
    DFS: find distinct factors > 12 with product = remaining.
    Returns (found, min_max) or (False, inf).
    """
    if _time_left() < 1:
        return False, float('inf')
    
    if remaining == 1:
        return True, max_so_far
    
    if len(chosen) >= 12:
        return False, float('inf')
    
    # We need at least one more factor
    # The remaining must be factorable into distinct factors > 12
    
    # Try each divisor of remaining that is > 12
    # Sort by value to try smaller factors first (to keep max low)
    divs = _divisors_gt(remaining, 13)
    
    best_max = float('inf')
    found = False
    
    for d in divs:
        if d in chosen:
            continue
        if d <= max_so_far and max_so_far > 0:
            # This doesn't help reduce max, but might be needed
            pass
        # Branch and bound: if d > best_max, skip
        if d >= best_max:
            continue
        if d > max_so_far:
            new_max = d
        else:
            new_max = max_so_far
        
        # Prune: if new_max >= best_max, skip
        if new_max >= best_max:
            continue
        
        # The remaining/ d must be factorable into distinct factors > 12
        # Also, remaining/d must be >= 13 (if we need more factors) or == 1
        rem2 = remaining // d
        if rem2 == 1:
            # This is the last factor
            if d > 12 and d not in chosen:
                if new_max < best_max:
                    best_max = new_max
                    found = True
        else:
            # Need to factor rem2 into more distinct factors > 12
            # rem2 must be >= 13 (at least one more factor)
            # Also, the minimum possible product of remaining factors
            # We need rem2 to have enough distinct divisors > 12
            sub_found, sub_max = _dfs(rem2, chosen | {d}, new_max, best_max)
            if sub_found and sub_max < best_max:
                best_max = sub_max
                found = True
    
    return found, best_max

def construct():
    target = factorial(12)  # 479001600
    
    # Try to find a decomposition minimizing the max factor
    # Start with a trivial solution: [target] with max = target
    best_max = target
    best_factors = [target]
    
    # DFS from the full product
    found, min_max = _dfs(target, set(), 0, best_max)
    
    if found:
        # We need to recover the actual factors, not just the max
        # Let's redo with factor tracking
        pass
    
    # Actually, let me restructure to track factors
    # For now, return the best we can
    # Let me do a cleaner version
    return _construct_with_tracking(target)

def _construct_with_tracking(target):
    best = {'max': target, 'factors': [target]}
    
    def dfs(remaining, chosen, current_factors, max_so_far):
        if _time_left() < 1:
            return
        
        if remaining == 1:
            if max_so_far < best['max']:
                best['max'] = max_so_far
                best['factors'] = list(current_factors)
            return
        
        if len(chosen) >= 12:
            return
        
        divs = _divisors_gt(remaining, 13)
        
        for d in divs:
            if d in chosen:
                continue
            if d >= best['max']:
                continue
            
            new_max = max(max_so_far, d)
            if new_max >= best['max']:
                continue
            
            rem2 = remaining // d
            if rem2 == 1:
                if new_max < best['max']:
                    best['max'] = new_max
                    best['factors'] = list(current_factors) + [d]
            else:
                dfs(rem2, chosen | {d}, current_factors + [d], new_max)
    
    dfs(target, set(), [], 0)
    
    return best['factors']
# EVOLVE-BLOCK-END
