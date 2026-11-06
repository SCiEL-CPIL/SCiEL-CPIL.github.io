import math
import time
from fractions import Fraction

def find_solution_backtrack(k, deadline):
    """
    Find k distinct denominators >= 5 summing to 1.
    Returns list of denominators or None.
    """
    target = Fraction(1, 1)
    
    # Precompute harmonic-like sums for quick max_possible checks
    # max_sum(start_d, n) = sum_{i=0}^{n-1} 1/(start_d + i)
    # For small n (<= 15) and moderate start_d, we can compute quickly
    
    def compute_max_sum(start_d, n):
        """Calculate max sum of 'n' distinct reciprocals starting from start_d."""
        if n <= 0:
            return Fraction(0, 1)
        s = Fraction(0, 1)
        for i in range(n):
            s += Fraction(1, start_d + i)
        return s
    
    def min_sum_from(start_d, n):
        """Calculate min sum of 'n' distinct reciprocals starting from start_d (largest denominators).
        This is 1/(start_d+n-1) + 1/(start_d+n) + ... but we need distinct and >= start_d.
        Actually for pruning, we need: can we achieve at least remaining_sum?
        The MAX possible sum uses smallest denominators: start_d, start_d+1, ...
        The MIN possible sum uses largest denominators, but we don't have an upper bound on d.
        So min_sum isn't directly useful for lower-bound pruning.
        Instead, use: if even the max possible sum < remaining, prune.
        """
        return Fraction(0, 1)
    
    def dfs(start_d, remaining_sum, terms_left, current_solution):
        if time.time() > deadline:
            return None
            
        if terms_left == 0:
            if remaining_sum == 0:
                return current_solution
            return None
            
        if terms_left == 1:
            # 1/d = remaining_sum => d = 1/remaining_sum
            if remaining_sum == 0:
                return None
            # d must be integer: remaining_sum = 1/d => d = 1/remaining_sum
            # Check if remaining_sum is a unit fraction
            num = remaining_sum.numerator
            den = remaining_sum.denominator
            if num == 1:
                d = den
                if d >= start_d and d <= 10000000:
                    return current_solution + [d]
            return None

        # Lower bound for d: 1/d <= remaining_sum => d >= 1/remaining_sum
        # Also d >= start_d
        # Use exact: d >= ceil(1/remaining_sum)
        # 1/remaining_sum = den/num where remaining_sum = num/den
        rs_num = remaining_sum.numerator
        rs_den = remaining_sum.denominator
        # d >= ceil(rs_den / rs_num)
        min_d_from_sum = (rs_den + rs_num - 1) // rs_num  # ceil division
        min_d = max(start_d, min_d_from_sum)
        
        # Upper bound for d:
        # We need to be able to sum terms_left reciprocals (starting from d+1) to get remaining_sum - 1/d.
        # The MAX sum of terms_left terms starting from d+1 is sum_{i=0}^{terms_left-1} 1/(d+1+i).
        # This must be >= remaining_sum - 1/d.
        # For a quick upper bound: even 1/(d+1) * terms_left >= remaining_sum - 1/d
        # So terms_left/(d+1) >= remaining_sum - 1/d
        # This gives a loose bound. Better: use the actual max sum check in the loop.
        
        # Also, d cannot be so large that even the smallest possible sum with remaining terms can't work.
        # But we don't have an upper bound on d, so we rely on the max_sum check.
        
        # Another constraint: the sum of the remaining terms_left-1 terms (starting from d+1)
        # must be at least remaining_sum - 1/d.
        # The max such sum is compute_max_sum(d+1, terms_left-1).
        # If compute_max_sum(d+1, terms_left-1) < remaining_sum - 1/d, then this d is too small?
        # No, if d is small, 1/d is large, so remaining_sum - 1/d is smaller, easier to achieve.
        # If d is large, 1/d is small, remaining_sum - 1/d is close to remaining_sum, harder to achieve.
        # So as d increases, it gets harder. So if for some d the max possible < needed, 
        # then for all larger d it's also impossible. So we can break.
        
        # Let's iterate d from min_d upward.
        # For each d, check if it's feasible.
        
        # Upper bound on d: 
        # We need remaining_sum - 1/d > 0, so 1/d < remaining_sum, d > 1/remaining_sum.
        # Also, the sum of terms_left-1 terms starting from d+1 must be >= remaining_sum - 1/d.
        # The max such sum decreases as d increases.
        # So there's a max d beyond which it's impossible.
        
        # Let's find max_d by checking when max_sum(d+1, terms_left-1) < remaining_sum - 1/d.
        # This is monotonic: as d increases, LHS decreases, RHS increases (since 1/d decreases).
        # So once it fails, it fails for all larger d.
        
        # To avoid iterating too much, estimate max_d.
        # Approx: terms_left/(d+1) >= remaining_sum - 1/d
        # For large d, 1/d ~ 0, so terms_left/d >= remaining_sum => d <= terms_left/remaining_sum
        max_d_est = int(float(terms_left) / float(remaining_sum)) + 2
        # But we also need d such that 1/d < remaining_sum, so d > 1/remaining_sum.
        # min_d already handles that.
        
        # Cap max_d to avoid huge ranges
        max_d = max(min_d, min(max_d_est, min_d + 100000))
        
        for d in range(min_d, max_d + 1):
            if time.time() > deadline:
                return None
                
            # Check 1/d < remaining_sum (strict, since we need positive remainder)
            if d * rs_num <= rs_den:
                # 1/d >= remaining_sum, so 1/d - remaining_sum >= 0, meaning remaining_sum - 1/d <= 0
                # We need remaining_sum - 1/d > 0 for the rest to sum to a positive value
                # Actually if remaining_sum - 1/d == 0 and terms_left > 1, we can't have positive denominators sum to 0.
                # So we need remaining_sum - 1/d > 0.
                continue
            
            next_remaining = remaining_sum - Fraction(1, d)
            if next_remaining <= 0:
                continue
                
            # Check max possible sum for remaining terms
            if terms_left - 1 > 0:
                max_possible = compute_max_sum(d + 1, terms_left - 1)
                if max_possible < next_remaining:
                    # Even with smallest possible denominators, we can't reach the sum
                    # Increasing d will make max_possible even smaller and next_remaining larger (since 1/d smaller)
                    # So break
                    break
            
            # Also check: can we achieve at least next_remaining?
            # The max is checked. The min isn't bounded above, so any d that passes max check is potentially feasible.
            
            # Recurse
            res = dfs(d + 1, next_remaining, terms_left - 1, current_solution + [d])
            if res is not None:
                return res
                
        return None

    # Start search
    # First denominator d1.
    # Max sum of k-1 terms starting from d1+1 must be >= 1 - 1/d1
    # Min d1 is 5.
    
    # Find max d1: 1/d1 < 1 (always true for d1 >= 2), and 
    # max_sum(d1+1, k-1) >= 1 - 1/d1
    # As d1 increases, 1 - 1/d1 increases, and max_sum decreases.
    # So there's a max d1.
    
    # Estimate: (k-1)/(d1+1) >= 1 - 1/d1
    # For large d1, ~ (k-1)/d1 >= 1 => d1 <= k-1
    # So d1 is at most around k.
    
    for d1 in range(5, min(50, k + 5)):
        if time.time() > deadline:
            return None
        next_remaining = 1 - Fraction(1, d1)
        if next_remaining <= 0:
            continue
        
        # Check max possible sum for remaining k-1 terms
        if k - 1 > 0:
            max_possible = compute_max_sum(d1 + 1, k - 1)
            if max_possible < next_remaining:
                continue
        
        res = dfs(d1 + 1, next_remaining, k - 1, [d1])
        if res is not None:
            return res
            
    return None

def construct():
    start_time = time.time()
    # 90s limit. Use 80s for safety.
    deadline = start_time + 80.0
    
    # Known solution k=10
    best_sol = [5, 6, 7, 8, 9, 10, 14, 15, 63, 2520]
    best_k = 10
    
    # Try k=8
    # Backtracking search
    sol_8 = find_solution_backtrack(8, deadline)
    if sol_8 is not None:
        return sol_8
        
    # Try k=9
    if time.time() < deadline - 5.0:
        sol_9 = find_solution_backtrack(9, deadline)
        if sol_9 is not None:
            return sol_9
            
    # Fallback to k=10
    return best_sol

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
