import time
from fractions import Fraction
from math import isqrt

TARGET = Fraction(5, 31)
MAX_DENOM = 10_000_000
MAX_TERMS = 256

def _get_divisors(n):
    """Generate all divisors of n."""
    if n == 0:
        return []
    divisors = set()
    for i in range(1, isqrt(n) + 1):
        if n % i == 0:
            divisors.add(i)
            divisors.add(n // i)
    return divisors

def _dfs_egyptian(remaining, min_denom, max_denom, count, used, path, deadline):
    """
    Recursively find 'count' distinct denominators in [min_denom, max_denom)
    that sum to 'remaining'.
    Returns a list of denominators or None.
    """
    if time.time() > deadline:
        return None
    
    if count == 0:
        if remaining == 0:
            return path[:]
        return None
    
    if count == 1:
        # 1/d = remaining => d = 1/remaining
        if remaining.numerator != 1:
            return None
        d = remaining.denominator
        if min_denom <= d < max_denom and d not in used:
            return path + [d]
        return None
    
    # Pruning:
    # Max possible sum with 'count' terms starting from min_denom
    # is sum(1/(min_denom + i) for i in range(count))
    # If this is < remaining, we need smaller denominators.
    # But min_denom is fixed for this call.
    # Actually, we iterate d from min_denom upwards.
    
    # We need to pick d such that:
    # 1. 1/d <= remaining  => d >= 1/remaining
    # 2. The remaining sum (remaining - 1/d) must be achievable with count-1 terms
    #    from denominators > d and < max_denom.
    
    # Lower bound for d:
    # d >= ceil(1/remaining)
    # Also d >= min_denom
    d_min = max(min_denom, int(1 / remaining) + (0 if 1 % remaining.denominator == 0 else 1))
    # Actually, 1/d <= remaining => d >= 1/remaining.
    # If 1/remaining is integer, d can be that.
    # Let's use Fraction comparison to be safe.
    # d >= 1/remaining  <=>  remaining >= 1/d
    # Let's just start from min_denom and check.
    
    # Upper bound for d:
    # We need to be able to sum 'remaining' with 'count' terms.
    # The max sum with 'count' terms all >= d is S_max(d) = sum(1/(d+i) for i in range(count)).
    # If S_max(d) < remaining, then even the smallest denominators (d, d+1, ...) are not enough.
    # So we must have smaller d.
    # Thus, if S_max(d) < remaining, we can break the loop (since d is increasing).
    
    # Also, we need the min sum with 'count' terms < max_denom to be <= remaining.
    # Min sum is sum(1/(max_denom - 1 - i) for i in range(count)).
    # If remaining < min_sum, return None (impossible for this state).
    
    min_sum = sum(Fraction(1, max_denom - 1 - i) for i in range(count))
    if remaining < min_sum:
        return None
    
    for d in range(d_min, max_denom):
        if time.time() > deadline:
            return None
            
        # Check if 1/d > remaining
        if Fraction(1, d) > remaining:
            continue
            
        # Check if max sum for remaining count-1 terms is sufficient
        # We pick d, so next min denom is d+1.
        # We need to sum rem2 = remaining - 1/d with count-1 terms from [d+1, max_denom).
        
        # Pruning check for the recursive call:
        # Max sum for count-1 terms starting from d+1:
        # S_max_next = sum(1/(d+1+i) for i in range(count-1))
        # If rem2 > S_max_next, then we can't reach rem2.
        # As d increases, rem2 decreases (1/d decreases) and S_max_next decreases.
        # This relationship is complex.
        
        # Simpler pruning:
        # If rem2 is too large for the available denominators.
        # Max sum of count-1 terms from [d+1, max_denom) is approx (count-1)/(d+1).
        # If rem2 > (count-1)/(d+1), break?
        # No, 1/(d+1) + ... is less than (count-1)/(d+1).
        # Let's use the exact max sum bound.
        
        # Calculate S_max_next
        # To avoid O(count) calculation every step, we can approximate or precompute?
        # Count is small.
        
        rem2 = remaining - Fraction(1, d)
        if rem2 < 0:
             # This shouldn't happen due to check above, but just in case
             continue
             
        # Check if rem2 is achievable
        # Min sum for count-1 terms from [d+1, max_denom)
        # If count-1 == 0, handled by base case.
        
        # Max sum for count-1 terms from [d+1, max_denom)
        # The largest possible sum is using the smallest available denominators: d+1, d+2, ...
        # S_max_next = sum(Fraction(1, d+1+i) for i in range(count-1))
        
        # If rem2 > S_max_next, we need smaller denominators.
        # Since d is increasing, the available denominators are larger, so S_max_next decreases.
        # So if rem2 > S_max_next, we can break.
        
        # Wait, rem2 also changes.
        # remaining is fixed. rem2 = remaining - 1/d.
        # As d increases, 1/d decreases, so rem2 increases.
        # S_max_next decreases.
        # So the gap (rem2 - S_max_next) increases.
        # So if rem2 > S_max_next, it will stay true for larger d.
        # So we can break.
        
        if count - 1 > 0:
             S_max_next = sum(Fraction(1, d+1+i) for i in range(count-1))
             if rem2 > S_max_next:
                 break
             
             # Also check min sum
             # Min sum for count-1 terms from [d+1, max_denom)
             # Uses largest available: max_denom-1, max_denom-2, ...
             # S_min_next = sum(Fraction(1, max_denom-1-i) for i in range(count-1))
             # If rem2 < S_min_next, we can't form it.
             # But we already checked min_sum for the whole 'count' at the start?
             # No, that was for the current state.
             # We should check if rem2 is too small.
             # If rem2 < S_min_next, return None?
             # No, we might pick larger d, which increases rem2.
             # So if rem2 < S_min_next, we should continue to larger d.
             
        # Recurse
        path.append(d)
        used.add(d)
        res = _dfs_egyptian(rem2, d + 1, max_denom, count - 1, used, path, deadline)
        if res is not None:
            return res
        path.pop()
        used.remove(d)
        
    return None

def _find_solution_for_max(M):
    """
    Try to find a set of distinct denominators in [1, M] that sum to TARGET.
    Returns a list of denominators if found, else None.
    """
    # Try 2, 3, 4 terms first as they are fast
    # 2-term
    num = 31 * M
    den = 5 * M - 31
    if den > 0 and num % den == 0:
        a = num // den
        if 1 <= a < M:
            return [a, M]
            
    # 3-term
    R_3 = TARGET - Fraction(1, M)
    if R_3 > 0:
        a_min = int(1 / R_3) + 1
        a_max = min(M - 2, int(2 / R_3))
        for a in range(a_min, a_max + 1):
            R_ab = R_3 - Fraction(1, a)
            if R_ab <= 0:
                continue
            if R_ab.denominator == 1:
                b = R_ab.numerator
                if b > a and b < M:
                    return [a, b, M]
            else:
                if R_ab.denominator % R_ab.numerator == 0:
                    b = R_ab.denominator // R_ab.numerator
                    if b > a and b < M:
                        return [a, b, M]

    # 4-term and above using DFS
    # We try k=4, 5, ...
    # Limit k to avoid explosion? 
    # For M < 248, k will be small.
    
    deadline = time.time() + 1.0 # 1 second per M? 
    # Better: global deadline.
    
    return None # Placeholder, will be replaced by construct logic

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # 50s margin
    
    # Known solution
    known = [8, 31, 248]
    best = known
    best_max = 248
    
    # Iterate M downwards
    for M in range(247, 1, -1):
        if time.time() > deadline:
            break
            
        # Try increasing number of terms
        # Max terms such that min possible sum > TARGET?
        # Min sum with k terms < M is approx k/M.
        # We need k/M <= 5/31 => k <= 5M/31.
        # Max k is roughly 5M/31.
        # Also k <= 256.
        
        max_k = min(256, int(5 * M / 31) + 1)
        
        found = False
        for k in range(2, max_k + 1):
            if time.time() > deadline:
                break
                
            # We want k-1 terms summing to R = 5/31 - 1/M
            R = TARGET - Fraction(1, M)
            if R <= 0:
                break # No more terms possible if R <= 0
            
            # Use DFS to find k-1 terms
            # min_denom = 1, max_denom = M
            # count = k-1
            # used = set()
            # path = []
            
            # Optimization: 
            # If k-1 == 1, handle directly.
            # If k-1 == 2, handle directly (2-term search for R).
            # Else use DFS.
            
            if k - 1 == 1:
                # 1/a = R
                if R.numerator == 1:
                    a = R.denominator
                    if 1 <= a < M:
                        return [a, M]
            elif k - 1 == 2:
                # 1/a + 1/b = R
                # a < b < M
                # Iterate a
                a_min = int(1 / R) + 1
                a_max = min(M - 2, int(2 / R))
                for a in range(a_min, a_max + 1):
                    R_ab = R - Fraction(1, a)
                    if R_ab <= 0:
                        continue
                    if R_ab.denominator == 1:
                        b = R_ab.numerator
                        if b > a and b < M:
                            return [a, b, M]
                    else:
                        if R_ab.denominator % R_ab.numerator == 0:
                            b = R_ab.denominator // R_ab.numerator
                            if b > a and b < M:
                                return [a, b, M]
            else:
                # DFS for k-1 terms
                res = _dfs_egyptian(R, 1, M, k - 1, set(), [], deadline)
                if res is not None:
                    return res + [M]
                    
    return best
