"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import gcd
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s margin for safety
    
    # Search for minimal k
    for k in range(8, 16):
        if time.time() > deadline - 5:
            break
        result = _search(k, deadline)
        if result is not None:
            return result
    return _fallback()

def _search(k, deadline):
    """Search for k distinct denominators >= 5 summing to 1 using integer arithmetic."""
    
    def max_sum_frac(remaining, start_d):
        """Return (num, den) for sum of 1/(start_d+i) for i in range(remaining)."""
        if remaining == 0:
            return (0, 1)
        num, den = 0, 1
        for i in range(remaining):
            d = start_d + i
            num = num * d + den
            den = den * d
        g = gcd(num, den)
        return (num // g, den // g)
    
    def dfs(idx, num, den, last_denom, chosen):
        """Find k-idx denominators > last_denom, each >= 5, summing to num/den."""
        if time.time() > deadline:
            return False
        
        remaining = k - idx
        
        if remaining == 0:
            return num == 0
        
        if num <= 0 or num > den:
            return False
        
        # Prune: max possible sum of remaining terms with smallest denominators
        m_num, m_den = max_sum_frac(remaining, last_denom + 1)
        if num * m_den > m_num * den:
            return False
        
        if remaining == 1:
            if den % num == 0:
                d_val = den // num
                if d_val > last_denom and d_val >= 5 and d_val <= 10000000:
                    chosen.append(d_val)
                    if dfs(idx + 1, 0, 1, d_val, chosen):
                        return True
                    chosen.pop()
            return False
        
        # Bounds for next denominator
        # 1/d < num/den strictly (since we need more positive terms)
        # So d > den/num
        min_d = max(last_denom + 1, den // num + 1)
        
        # Upper bound: d <= remaining * den // num (loose but safe)
        max_d = min(10000000, remaining * den // num + 2)
        
        for d_val in range(min_d, max_d + 1):
            if time.time() > deadline:
                return False
            
            # 1/d_val must be < num/den
            if d_val * num <= den:
                continue
            
            new_num = num * d_val - den
            new_den = den * d_val
            
            if new_num <= 0:
                continue
            
            g = gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g
            
            chosen.append(d_val)
            if dfs(idx + 1, new_num, new_den, d_val, chosen):
                return True
            chosen.pop()
        
        return False
    
    chosen = []
    if dfs(0, 1, 1, 4, chosen):
        return chosen
    return None

def _fallback():
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
