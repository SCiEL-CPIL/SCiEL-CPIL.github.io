import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    # Known solution
    best_solution = [8, 31, 248]
    
    # Target sum is 5/31
    target_total = Fraction(5, 31)
    
    # Search for solutions with M < 248
    # M is the largest denominator.
    # We need sum of 1/a_i = 5/31 with a_i < M and one a_i = M? 
    # No, the problem says "Return 1..256 distinct denominators ... Minimize the LARGEST denominator".
    # So if we find a set with max denominator M, that's valid.
    # We can iterate M from a lower bound to 247.
    # For a fixed M, we must include M in the set? 
    # If we are looking for a set with max denominator EXACTLY M, then M must be in the set.
    # If we find a set with max denominator < M, we would have found it in a previous iteration.
    # So yes, for each M, we assume M is in the set, and we need to find other denominators < M 
    # that sum to 5/31 - 1/M.
    
    for M in range(1, 248):
        if time.time() > deadline:
            break
            
        # If 1/M >= 5/31, we can't even use M alone, and we need positive terms, so sum > 1/M.
        # Actually, sum = 5/31. If 1/M > 5/31, then we can't use M as the largest term 
        # because the sum of other positive terms would make the total > 1/M > 5/31? 
        # No, if 1/M > 5/31, then M < 31/5 = 6.2. 
        # If M=6, 1/6 = 0.166, 5/31 = 0.161. 1/6 > 5/31.
        # Then we need sum of other terms = 5/31 - 1/6 = (30-31)/186 = -1/186. Impossible.
        # So M must be such that 1/M < 5/31, i.e., M > 6.2. So M >= 7.
        
        if M < 7:
            continue
            
        remaining = target_total - Fraction(1, M)
        if remaining <= 0:
            continue
            
        # We need to find distinct integers a_1, ..., a_k < M 
        # such that sum(1/a_i) = remaining.
        # And all a_i must be distinct and not equal to M (already ensured by < M).
        
        solution = _find_subset(remaining, 1, M - 1, M, deadline)
        if solution is not None:
            return solution
            
    return best_solution

def _find_subset(remaining, min_denom, max_denom, M, deadline):
    """
    Finds a subset of {min_denom, ..., max_denom} that sums to 'remaining'.
    Returns the list of denominators or None.
    """
    if remaining == 0:
        return []
    
    if remaining < 0:
        return None
        
    # Pruning: If the sum of all possible reciprocals from min_denom to max_denom is less than remaining, impossible.
    # Sum_{i=min_denom}^{max_denom} 1/i
    # We can approximate or calculate exactly. Given small M, exact is fine.
    # But calculating sum of fractions repeatedly is expensive.
    # Let's use a simple bound: max_possible_sum = (max_denom - min_denom + 1) / min_denom
    # This is a loose upper bound.
    # A tighter bound: H(max_denom) - H(min_denom - 1)
    # Let's just try the DFS and rely on the start index pruning.
    
    # We must pick at least one number.
    # The smallest number we can pick is d_min = ceil(1/remaining)
    # Also d_min >= min_denom
    
    d_min = max(min_denom, math.ceil(1 / float(remaining)))
    
    # If d_min > max_denom, no solution
    if d_min > max_denom:
        return None
        
    # Pruning: Check if sum of reciprocals from d_min to max_denom is sufficient.
    # If not, return None.
    # sum_{i=d_min}^{max_denom} 1/i
    # We can compute this exactly or approximately.
    # Approximation: log(max_denom/d_min)
    # Let's use a quick exact sum check for small ranges, or just skip this heavy pruning 
    # and rely on the recursive structure. The range is small (< 248).
    
    for d in range(d_min, max_denom + 1):
        if time.time() > deadline:
            return None
            
        # If 1/d > remaining, we can break? 
        # No, if 1/d > remaining, then for larger d, 1/d is smaller. 
        # But we started at ceil(1/remaining), so 1/d <= remaining for all d >= d_min.
        
        new_remaining = remaining - Fraction(1, d)
        
        if new_remaining < 0:
            # Since d is increasing, 1/d is decreasing. 
            # If 1/d > remaining, then for smaller d it was even larger. 
            # But we started at d_min where 1/d_min <= remaining.
            # So this shouldn't happen unless float error in d_min calc? 
            # We use Fraction, so it's exact.
            # If new_remaining < 0, it means 1/d > remaining.
            # But we chose d >= ceil(1/remaining). 
            # If remaining is not integer, ceil(1/remaining) ensures 1/d <= remaining?
            # Example: remaining = 0.4. 1/0.4 = 2.5. ceil = 3. 1/3 = 0.33 < 0.4.
            # Example: remaining = 0.2. 1/0.2 = 5. ceil = 5. 1/5 = 0.2.
            # So new_remaining should be >= 0.
            continue
            
        # Recurse
        # The next denominator must be > d
        result = _find_subset(new_remaining, d + 1, max_denom, M, deadline)
        
        if result is not None:
            return [d] + result
            
    return None
