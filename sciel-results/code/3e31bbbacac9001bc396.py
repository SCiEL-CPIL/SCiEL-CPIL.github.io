import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    # Try to find a solution with 8 terms
    result = _search_k_list(8, deadline)
    if result:
        return result
    
    # Fallback to larger k if 8 fails
    for k in range(9, 13):
        if time.time() > deadline - 5:
            break
        result = _search_k_list(k, deadline)
        if result:
            return result
    
    # Last resort: return a hardcoded valid solution if available
    # We don't have a hardcoded one, so we rely on the search.
    # If search fails, it's a critical error.
    raise RuntimeError("no valid representation found")

def _max_sum_float(m, start):
    """Approximate max sum of m unit fractions with denominators >= start"""
    s = 0.0
    for i in range(m):
        s += 1.0 / (start + i)
    return s

def _search_k_list(k, deadline):
    """Search for k distinct denominators >= 5 summing to 1, returning the list"""
    d = []
    
    def dfs(idx, current_sum_float, last_denom):
        if time.time() > deadline:
            return None
        
        if idx == k:
            # Verify with Fraction
            total = Fraction(0)
            for val in d:
                total += Fraction(1, val)
            if total == 1:
                return d[:]
            return None
        
        remaining_terms = k - idx
        remaining_float = 1.0 - current_sum_float
        
        if remaining_float <= 0:
            return None
        
        # Pruning: max possible sum of remaining terms
        max_possible_float = _max_sum_float(remaining_terms, last_denom + 1)
        if current_sum_float + max_possible_float < 1.0 - 1e-12:
            return None
        
        # Determine range for next denominator
        # 1/d <= remaining => d >= 1/remaining
        # Add small epsilon to avoid rounding up unnecessarily
        min_d_float = 1.0 / remaining_float
        min_d = max(last_denom + 1, int(math.ceil(min_d_float - 1e-10)))
        
        # Upper bound: d <= remaining_terms / remaining
        # Add small epsilon to avoid rounding down unnecessarily
        max_d_float = remaining_terms / remaining_float
        max_d = int(math.floor(max_d_float + 1e-10))
        
        if min_d > max_d:
            return None
        
        # Special case: last term
        if remaining_terms == 1:
            d_val = 1.0 / remaining_float
            d_int = int(round(d_val))
            if abs(d_val - d_int) < 1e-10 and d_int > last_denom and d_int <= 10000000:
                d.append(d_int)
                # Verify exact sum
                total = Fraction(0)
                for val in d:
                    total += Fraction(1, val)
                if total == 1:
                    return d[:]
                d.pop()
            return None
        
        # Iterate possible denominators
        for d_val in range(min_d, max_d + 1):
            if time.time() > deadline:
                return None
            
            new_sum_float = current_sum_float + 1.0 / d_val
            if new_sum_float > 1.0:
                continue
            
            d.append(d_val)
            res = dfs(idx + 1, new_sum_float, d_val)
            if res:
                return res
            d.pop()
        
        return None
    
    return dfs(0, 0.0, 4)
