import time
from fractions import Fraction
import math

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin before 60s kill (evaluator kills at 90s, but let's be safe)
    
    target = Fraction(5, 31)
    
    # Known baseline solution
    best_solution = [8, 28, 1736]
    best_max_den = 1736
    
    # Precompute harmonic numbers approximation for pruning
    # H_n = ln(n) + gamma + 1/(2n) - 1/(12n^2)
    gamma = 0.5772156649015329
    
    def harmonic_approx(n):
        if n <= 0:
            return 0.0
        return math.log(n) + gamma + 1.0/(2.0*n) - 1.0/(12.0*n*n)
        
    def can_complete(remainder, min_denom, max_denom):
        """
        Check if it's possible to sum to 'remainder' using distinct denominators 
        in (min_denom, max_denom].
        """
        if remainder == 0:
            return True
        if remainder < 0:
            return False
        if max_denom <= min_denom:
            return False
            
        # The maximum sum we can achieve is the sum of 1/d for all d in (min_denom, max_denom]
        # This is H_max - H_min
        # If this max sum is less than the remainder, it's impossible.
        max_sum = harmonic_approx(max_denom) - harmonic_approx(min_denom)
        
        # Use a small epsilon for floating point comparison
        if max_sum < float(remainder) - 1e-12:
            return False
            
        # Additional check: The minimum denominator we *must* pick is at least ceil(1/remainder)
        # because 1/d <= remainder => d >= 1/remainder
        # If min_denom + 1 is smaller than this, we effectively start from ceil(1/remainder)
        # But the harmonic bound above already accounts for all d > min_denom.
        # A tighter bound: we cannot use any d < ceil(1/remainder).
        # Let d_min_required = max(min_denom + 1, ceil(1/remainder))
        # If d_min_required > max_denom, return False
        # The max sum is then H_max - H_{d_min_required - 1}
        
        # Calculate d_min_required
        # ceil(1/remainder) = (1/remainder).numerator // (1/remainder).denominator + 1 if not integer
        inv_rem = 1 / remainder
        d_min_req = math.ceil(inv_rem)
        
        d_start = max(min_denom + 1, d_min_req)
        
        if d_start > max_denom:
            return False
            
        # Re-calculate max sum with the tighter lower bound
        max_sum_tight = harmonic_approx(max_denom) - harmonic_approx(d_start - 1)
        if max_sum_tight < float(remainder) - 1e-12:
            return False
            
        return True

    def dfs(remainder, min_denom, max_denom, current_set):
        """
        Returns a list of denominators if a solution is found, else None.
        """
        if time.time() - start_time > time_limit:
            return None
            
        if remainder == 0:
            return current_set
            
        if remainder < 0:
            return None
            
        # Pruning
        if not can_complete(remainder, min_denom, max_denom):
            return None
            
        # Determine the range of next denominators
        # d must be > min_denom
        # d must be <= max_denom
        # 1/d <= remainder => d >= 1/remainder
        d_min = max(min_denom + 1, math.ceil(1 / remainder))
        d_max = max_denom
        
        if d_min > d_max:
            return None
            
        # Heuristic: Try smaller d first? 
        # Smaller d means larger 1/d, reducing remainder faster.
        # This is often good for finding *a* solution.
        # For minimizing max denom, we are already constrained by max_denom.
        
        for d in range(d_min, d_max + 1):
            if time.time() - start_time > time_limit:
                return None
                
            next_rem = remainder - Fraction(1, d)
            
            if next_rem < 0:
                # 1/d > remainder. Since d is increasing, 1/d is decreasing.
                # So next_rem will increase. We can continue.
                continue
                
            if next_rem == 0:
                return current_set + [d]
                
            # Recurse
            # Next min denom is d (to ensure distinctness and increasing order)
            result = dfs(next_rem, d, max_denom, current_set + [d])
            if result is not None:
                return result
                
        return None

    # Binary search for the minimum K such that a solution exists with max denom <= K
    # We know K=1736 works.
    # Let's check if smaller K works.
    
    low = 1
    high = best_max_den
    
    # We can optimize by checking a few specific values first to narrow down?
    # Or just binary search.
    # Binary search assumes monotonicity: if solution exists for K, it exists for K+1.
    # This is true because the set of valid denominators for K is a subset of those for K+1.
    
    # However, DFS might be slow for infeasible K.
    # To mitigate this, we can add a "quick greedy check" before DFS?
    # Or just rely on pruning.
    
    # Let's try binary search.
    # To be safe, we keep the known best solution.
    
    # Optimization: If the range is small, linear search down might be better?
    # 1736 is not huge. Log2(1736) is ~11.
    
    # We will iterate through midpoints.
    # If a solution is found for K, we update best and set high = K - 1.
    # If no solution is found, we set low = K + 1.
    # If we run out of time, we return the best found.
    
    # Initial check: Is there a solution with K < 1736?
    # We'll do the binary search.
    
    while low <= high and time.time() - start_time < time_limit:
        mid = (low + high) // 2
        
        # Try to find a solution with max denom <= mid
        # We pass an empty current set
        solution = dfs(target, 0, mid, [])
        
        if solution is not None:
            # Found a solution with max <= mid
            actual_max = max(solution)
            if actual_max < best_max_den:
                best_max_den = actual_max
                best_solution = solution
            # Try to find an even smaller max
            high = mid - 1
        else:
            # No solution found for mid (or timeout)
            # If timeout, we don't know for sure, but we can assume infeasible for this search path?
            # Actually, if we timeout, we should probably stop the binary search to avoid wasting time.
            if time.time() - start_time > time_limit:
                break
            low = mid + 1
            
    # If we exhausted the search or ran out of time, return the best solution found
    # The best solution is guaranteed to be valid because we only update it with valid solutions from DFS.
    
    return best_solution
