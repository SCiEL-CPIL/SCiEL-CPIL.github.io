"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def construct():
    """
    Find minimum number of distinct integer denominators in [5, 10^7] 
    whose reciprocal sum is exactly 1.
    Uses split-one identity: 1/n = 1/(n+1) + 1/(n*(n+1))
    """
    deadline = time.time() + 55  # Leave 35s safety margin
    
    # Known valid solutions as fallbacks
    fallbacks = [
        [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    ]
    
    # Verify and keep only valid fallbacks
    valid_fallbacks = []
    for fb in fallbacks:
        num, den = _sum_reciprocals(fb)
        if num == den and all(5 <= d <= 10000000 for d in fb) and len(set(fb)) == len(fb):
            valid_fallbacks.append(fb)
    
    # Try to find solutions with increasing term counts
    for k in range(5, 20):
        if time.time() > deadline - 10:
            break
        result = _split_one_search(k, deadline)
        if result is not None:
            num, den = _sum_reciprocals(result)
            if num == den and len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                return result
    
    # Use best valid fallback
    if valid_fallbacks:
        # Return the one with fewest terms
        valid_fallbacks.sort(key=len)
        return valid_fallbacks[0]
    
    raise RuntimeError("no representation found")

def _sum_reciprocals(denoms):
    """Compute sum of 1/d for d in denoms as (num, den) in reduced form."""
    num, den = 0, 1
    for d in denoms:
        # num/den + 1/d = (num*d + den) / (den*d)
        num, den = num * d + den, den * d
        g = gcd(num, den)
        num //= g
        den //= g
    return num, den

def _max_sum(min_d, k):
    """Upper bound: sum of k largest reciprocals starting from min_d"""
    total = 0.0
    for i in range(k):
        total += 1.0 / (min_d + i)
    return total

def _split_one_search(k, deadline):
    """
    Search using split-one identity: 1/n = 1/(n+1) + 1/(n*(n+1))
    Start from a base representation and split terms to increase count.
    """
    # Start with minimal representation and try to split
    # Base: try to find a small set that sums to 1, then split terms
    
    # Alternative: direct backtracking with split-one inspired pruning
    def search(depth, min_d, rem_num, rem_den, used_set):
        """
        Find (k - depth) denominators >= min_d such that their reciprocal sum = rem_num/rem_den.
        """
        if time.time() > deadline:
            return None
        
        if depth == k:
            if rem_num == 1:
                d = rem_den
                if d >= min_d and d >= 5 and d <= 10000000 and d not in used_set:
                    return [d]
            return None
        
        remaining_terms = k - depth
        
        # Pruning: max possible sum
        if _max_sum(min_d, remaining_terms) < rem_num / rem_den:
            return None
        
        if rem_num == 0:
            return None
        
        # Lower bound: d > rem_den / rem_num
        lower = max(min_d, rem_den // rem_num + 1)
        
        # Upper bound with tighter constraint
        # We need rem_num/rem_den - 1/d to be representable by (remaining_terms-1) terms > d
        # So 1/d >= rem_num/rem_den - _max_sum(d+1, remaining_terms-1)
        # For efficiency, limit range
        upper = min(10000000, lower + 50000)
        
        for d in range(lower, upper + 1):
            if time.time() > deadline:
                return None
            
            if d in used_set:
                continue
            
            # 1/d must be < rem_num/rem_den
            if d * rem_num <= rem_den:
                continue
            
            new_num = rem_num * d - rem_den
            new_den = rem_den * d
            
            if new_num <= 0:
                continue
            
            g = gcd(new_num, new_den)
            new_num //= g
            new_den //= g
            
            # Pruning
            if remaining_terms - 1 > 0:
                if _max_sum(d + 1, remaining_terms - 1) < new_num / new_den:
                    break
            
            new_used = used_set | {d}
            result = search(depth + 1, d + 1, new_num, new_den, new_used)
            if result is not None:
                return [d] + result
        
        return None
    
    # Try different starting points
    for start_min in [5, 6, 7, 8]:
        if time.time() > deadline:
            break
        result = search(0, start_min, 1, 1, set())
        if result is not None:
            return result
    
    return None

# EVOLVE-BLOCK-END
