"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 90s limit, keep margin
    
    # Search for minimal k starting from 5
    for k in range(5, 10):
        if time.time() > deadline - 5:
            break
        result = _search(k, deadline)
        if result:
            return result
    # Fallback
    return _fallback()

def _search(k, deadline):
    """Find k distinct denominators in [5, 10^7] with reciprocal sum = 1."""
    d = []
    
    def dfs(idx, num, den, last_denom):
        """
        num/den is the current sum of chosen reciprocals.
        We need to add (k - idx) more terms to reach 1.
        remaining = 1 - num/den = (den - num) / den
        """
        if time.time() > deadline:
            return None
        
        if idx == k:
            if num == den:
                return d[:]
            return None
        
        remaining_num = den - num
        if remaining_num <= 0:
            return None
        
        # Pruning: max possible sum of (k-idx) terms with denominators > last_denom
        # Max sum = sum_{i=1}^{k-idx} 1/(last_denom + i)
        # This is an upper bound; if current + max < 1, prune
        rem_terms = k - idx
        # Approximate max sum using integral or just first few terms
        # For tightness, compute exactly for small rem_terms
        if rem_terms <= 10:
            max_sum_num = 0
            max_sum_den = 1
            for i in range(rem_terms):
                dd = last_denom + 1 + i
                # Add 1/dd to max_sum
                new_num = max_sum_num * dd + max_sum_den
                new_den = max_sum_den * dd
                g = gcd(new_num, new_den)
                max_sum_num = new_num // g
                max_sum_den = new_den // g
            # Check if num/den + max_sum < 1
            # i.e., num/den + max_sum_num/max_sum_den < 1
            # i.e., num * max_sum_den + max_sum_num * den < den * max_sum_den
            if num * max_sum_den + max_sum_num * den < den * max_sum_den:
                return None
        else:
            # Cruder bound: sum of 1/(last_denom+1) to 1/(last_denom+rem_terms)
            # < rem_terms / (last_denom + 1)
            # If num/den + rem_terms/(last_denom+1) < 1, prune
            # i.e., num * (last_denom+1) + rem_terms * den < den * (last_denom+1)
            if num * (last_denom + 1) + rem_terms * den < den * (last_denom + 1):
                return None
        
        # Min denominator for next term:
        # 1/d >= remaining/(k-idx) is NOT required; we just need 1/d <= remaining
        # Actually 1/d <= remaining => d >= 1/remaining = den/remaining_num
        # Also d > last_denom
        min_d = max(last_denom + 1, den // remaining_num + 1)
        
        # If remaining_num == den (i.e., remaining = 1), then min_d >= 5
        # But last_denom starts at 4, so min_d >= 5
        
        # Upper bound for d:
        # We need 1/d + max_sum(rem_terms-1, d+1) >= remaining
        # Approx: 1/d + (rem_terms-1)/(d+1) >= remaining
        # For large d, this is approx rem_terms/d >= remaining
        # So d <= rem_terms / remaining = rem_terms * den / remaining_num
        max_d = min(10000000, rem_terms * den // remaining_num + 10)
        
        if min_d > max_d:
            return None
        
        # If only one term left, solve exactly
        if rem_terms == 1:
            # 1/d = remaining => d = den / remaining_num
            d_val = den // remaining_num
            if remaining_num * d_val == den and d_val > last_denom and d_val >= 5 and d_val <= 10000000:
                d.append(d_val)
                if dfs(idx + 1, den, den, d_val):
                    return d[:]
                d.pop()
            return None
        
        for d_val in range(min_d, max_d + 1):
            if time.time() > deadline:
                return None
            
            # new_sum = num/den + 1/d_val = (num * d_val + den) / (den * d_val)
            new_num = num * d_val + den
            new_den = den * d_val
            
            if new_num >= new_den:
                # Sum already >= 1, can't fix by adding more positive terms
                # Actually if new_num == new_den, we're done only if idx+1 == k
                if idx + 1 == k and new_num == new_den:
                    d.append(d_val)
                    return d[:]
                continue
            
            # Reduce fraction
            g = gcd(new_num, new_den)
            rn = new_num // g
            rd = new_den // g
            
            d.append(d_val)
            res = dfs(idx + 1, rn, rd, d_val)
            if res:
                return res
            d.pop()
        
        return None
    
    return dfs(0, 0, 1, 4)

def _fallback():
    # Known solution: 1/5 + 1/6 + 1/7 + 1/10 + 1/42 + 1/140 = ?
    # Let's use a verified solution
    # 1/5 + 1/6 + 1/7 + 1/10 + 1/42 + 1/140
    # LCM(5,6,7,10,42,140) = 420
    # 84 + 70 + 60 + 42 + 10 + 3 = 269 / 420 != 1
    # Try: 1/5 + 1/6 + 1/7 + 1/10 + 1/35 + 1/42
    # LCM = 210
    # 42 + 35 + 30 + 21 + 6 + 5 = 139/210
    # Use the seed fallback approach
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
