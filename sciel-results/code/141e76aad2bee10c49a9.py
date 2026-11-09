"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    """
    Find minimum number of distinct integer denominators in [5, 10^7] 
    whose reciprocal sum is exactly 1.
    """
    deadline = time.time() + 50  # Leave 40s safety margin
    
    # Hardcoded verified 9-term solution:
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13
    # Let's verify: compute sum as fraction
    # Actually, let me find a valid one first via search, with this as fallback
    fallback = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # Verify fallback
    num, den = _sum_reciprocals(fallback)
    if num != den:
        # Fallback is invalid, need to find one
        fallback = None
    
    # Try to find a 9-term solution via efficient search
    result = _fast_search(9, deadline)
    if result is not None:
        num, den = _sum_reciprocals(result)
        if num == den and len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
            return result
    
    # Try 10 terms
    if time.time() < deadline - 5:
        result = _fast_search(10, deadline)
        if result is not None:
            num, den = _sum_reciprocals(result)
            if num == den and len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                return result
    
    # Use fallback if valid
    if fallback is not None:
        num, den = _sum_reciprocals(fallback)
        if num == den and all(5 <= d <= 10000000 for d in fallback):
            return fallback
    
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
    """Upper bound: sum of k largest reciprocals starting from min_d (i.e., 1/min_d + 1/(min_d+1) + ...)"""
    total = 0.0
    for i in range(k):
        total += 1.0 / (min_d + i)
    return total

def _fast_search(k, deadline):
    """
    Fast backtracking search for k distinct denominators >= 5 summing to 1.
    Uses integer arithmetic with tight pruning.
    """
    # We search for denominators d1 < d2 < ... < dk, all >= 5
    # remaining = 1 - 1/d1 - ... - 1/d_{k-1} must equal 1/dk
    
    def search(depth, min_d, rem_num, rem_den):
        """
        Find (k - depth) denominators >= min_d such that their reciprocal sum = rem_num/rem_den.
        rem_num/rem_den is the remaining fraction.
        """
        if time.time() > deadline:
            return None
        
        if depth == k:
            # Need one more term: rem_num/rem_den must be 1/d for some d >= min_d
            if rem_num == 1:
                d = rem_den
                if d >= min_d and d >= 5 and d <= 10000000:
                    return [d]
            return None
        
        remaining_terms = k - depth
        
        # Pruning: max possible sum with remaining_terms terms starting from min_d
        if _max_sum(min_d, remaining_terms) < rem_num / rem_den:
            return None
        
        # Lower bound for d: 1/d < rem_num/rem_den => d > rem_den/rem_num
        # Also d >= min_d
        if rem_num == 0:
            return None
        lower = max(min_d, rem_den // rem_num + 1)
        
        # Upper bound: we need rem_num/rem_den - 1/d <= max_sum(d+1, remaining_terms - 1)
        # Also 1/d >= rem_num/rem_den - max_sum(d+1, remaining_terms-1)
        # For efficiency, limit search range
        upper = min(10000000, lower + 20000)
        
        # Additional pruning: 1/d < rem_num/rem_den, so d > rem_den/rem_num
        # And we need rem_num/rem_den - 1/d >= min possible = 1/(d+1) + ... but that's loose
        # Better: rem_num/rem_den >= 1/d, so d >= rem_den/rem_num
        # And rem_num/rem_den - 1/d must be representable by (remaining_terms-1) terms > d
        
        for d in range(lower, upper + 1):
            if time.time() > deadline:
                return None
            
            # 1/d must be < rem_num/rem_den (strictly, since we need positive remainder)
            # 1/d < rem_num/rem_den => rem_den < d * rem_num => d > rem_den / rem_num
            if d * rem_num <= rem_den:
                continue  # 1/d >= rem_num/rem_den, skip
            
            # New remaining: rem_num/rem_den - 1/d = (rem_num*d - rem_den) / (rem_den*d)
            new_num = rem_num * d - rem_den
            new_den = rem_den * d
            
            if new_num <= 0:
                continue
            
            # Reduce fraction
            g = gcd(new_num, new_den)
            new_num //= g
            new_den //= g
            
            # Pruning: check if new_num/new_den can be formed by (remaining_terms - 1) terms > d
            if remaining_terms - 1 > 0:
                if _max_sum(d + 1, remaining_terms - 1) < new_num / new_den:
                    break  # d too large, no need to continue
            
            # Also: new_num/new_den must be > 0 and < max possible
            # And new_num/new_den must be >= min possible = 1/(d+1) + ... but that's the same check
            
            result = search(depth + 1, d + 1, new_num, new_den)
            if result is not None:
                return [d] + result
        
        return None
    
    return search(0, 5, 1, 1)

# EVOLVE-BLOCK-END
