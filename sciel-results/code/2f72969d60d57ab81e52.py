import time
from fractions import Fraction
from functools import lru_cache

def construct():
    deadline = time.time() + 85  # Safety margin within 90s
    
    def dfs(remaining, min_d, count, path):
        """Find 'count' more distinct denominators >= min_d that sum to 'remaining'."""
        if time.time() > deadline:
            return None
        if count == 0:
            if remaining == 0:
                return list(path)
            return None
        if remaining <= 0:
            return None
        
        # Pruning: max possible sum with 'count' terms starting from min_d
        if remaining > Fraction(count, min_d):
            return None
        
        # Lower bound for d: 1/d <= remaining -> d >= 1/remaining
        d_start = max(min_d, (remaining.numerator + remaining.denominator - 1) // remaining.denominator)
        
        # Upper bound for d:
        # We need remaining - 1/d <= (count-1)/(d+1)
        # remaining <= 1/d + (count-1)/(d+1)
        # Let k = count - 1
        # remaining <= 1/d + k/(d+1)
        # The function f(d) = 1/d + k/(d+1) is decreasing for d > 0.
        # So we can find the max d such that f(d) >= remaining.
        # Approximate d_max: 1/d ~ remaining => d ~ 1/remaining.
        # Let's just iterate and break if the condition fails.
        # But we need an upper limit to avoid infinite loops or huge ranges if start is small.
        # d_start is at least 1/remaining.
        # d_max is roughly 1/remaining as well?
        # If k is small, 1/d dominates.
        
        # Let's iterate d from d_start upwards.
        # To prevent excessive iteration, we can estimate d_max.
        # 1/d + k/(d+1) >= remaining
        # For large d, LHS ~ (1+k)/d. So d <= (1+k)/remaining.
        # So d_max <= (count) / remaining.
        # This is a safe upper bound.
        
        d_end = int((count * remaining.numerator) // remaining.denominator) + 10
        d_end = min(d_end, 10000000)
        
        # If d_start > d_end, no solution
        if d_start > d_end:
            return None

        for d in range(d_start, d_end + 1):
            if time.time() > deadline:
                return None
            if d in path:
                continue
            
            # Check if 1/d <= remaining
            if Fraction(1, d) > remaining:
                # Since d is increasing, 1/d is decreasing.
                # If 1/d > remaining, we need larger d.
                # But we started at d_start which should satisfy this.
                # However, due to integer division, d_start might be slightly off.
                continue
            
            new_remaining = remaining - Fraction(1, d)
            
            if new_remaining < 0:
                continue
                
            # Feasibility check for remaining 'count-1' terms
            if count - 1 == 0:
                if new_remaining == 0:
                    return list(path) + [d]
                else:
                    continue
            
            # Max sum of (count-1) terms from d+1 is (count-1)/(d+1)
            if new_remaining > Fraction(count - 1, d + 1):
                # If new_remaining is too large, we need smaller d?
                # No, d is increasing. As d increases, 1/d decreases, so new_remaining increases.
                # Also (count-1)/(d+1) decreases.
                # So the gap widens.
                # If it fails here, it will fail for all larger d.
                break
            
            # Min sum of (count-1) terms from d+1 is (count-1)/10000000
            # if new_remaining < Fraction(count - 1, 10000000):
            #     # If new_remaining is too small, we need larger d?
            #     # As d increases, new_remaining increases.
            #     # So we continue.
            #     pass

            path.append(d)
            result = dfs(new_remaining, d + 1, count - 1, path)
            path.pop()
            if result is not None:
                return result
        
        return None
    
    # Try n from 3 upward
    for n in range(3, 10):
        if time.time() > deadline:
            break
        result = dfs(Fraction(1), 5, n, [])
        if result is not None:
            # Verify
            s = sum(Fraction(1, d) for d in result)
            if s == 1 and len(set(result)) == n and all(5 <= d <= 10000000 for d in result):
                return result
    
    # Fallback: DP over divisors of a large number or just return a known valid set
    # Since the search for n=3..9 should find a solution quickly, 
    # we provide a hardcoded fallback of a valid set of size 6 or so if search fails.
    # 1 = 1/5 + 1/6 + 1/7 + 1/42 is invalid.
    # Known valid: 1 = 1/5 + 1/6 + 1/8 + 1/120 + 1/1440 + 1/10080?
    # Let's just use the DP fallback from the original code but fixed.
    
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total in list(reached.keys()):
            new_total = total + w
            if new_total <= target and new_total not in reached:
                reached[new_total] = reached[total] + [d]
        if target in reached:
            sol = reached[target]
            s = sum(Fraction(1, d) for d in sol)
            if s == 1 and all(5 <= d <= 10000000 for d in sol) and len(set(sol)) == len(sol):
                return sol

    # Last resort: raise error or return empty? 
    # The task requires 1..256 denominators.
    # If we fail, we must raise or return invalid? 
    # "invalid result... scores 0".
    # But we should try to return *something* valid if possible.
    # A very large n solution is easy to find via greedy?
    # Greedy: 1 = 1/5 + 4/5. 4/5 = 1/6 + 14/30 = 1/6 + 7/15...
    # Let's just return a known valid solution if DFS fails.
    # 1 = 1/5 + 1/6 + 1/7 + 1/42 + 1/140 + 1/210 + 1/420 + 1/840 + 1/1680 + 1/3360 + 1/6720 + 1/13440 + 1/26880 + 1/53760 + 1/107520 + 1/215040 + 1/430080 + 1/860160 + 1/1720320 + 1/3440640 + 1/6881280 + 1/13762560 + 1/27525120 + 1/55050240 + 1/110100480 + 1/220200960 + 1/440401920 + 1/880803840 + 1/1761607680 + 1/3523215360 + 1/7046430720 + 1/14092861440 ...
    # This is getting long.
    
    # Actually, the DP on 2520 should find a solution.
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35?
    # 1/5+1/6+1/7+1/10+1/14+1/35 = (42+35+30+21+15+6)/210 = 149/210. No.
    
    # If all else fails, return an empty list? No, must be 1..256.
    # I will trust the DFS or DP.
    raise RuntimeError("no valid representation found")
