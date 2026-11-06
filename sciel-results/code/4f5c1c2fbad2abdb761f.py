from fractions import Fraction
import time

def construct():
    start_time = time.time()
    
    def dfs(remaining, k, min_d, current):
        if time.time() - start_time > 85:
            return None
            
        if k == 0:
            if remaining == 0:
                return current
            return None
            
        if remaining == 0:
            return None
            
        # Upper bound for d
        # Sum of k terms starting at d is approx k/d.
        # We need k/d >= remaining => d <= k/remaining
        max_d_approx = int(k / remaining)
        if max_d_approx > 10000000:
            max_d_approx = 10000000
            
        d = min_d
        while d <= max_d_approx:
            term = Fraction(1, d)
            if term >= remaining:
                break
                
            new_rem = remaining - term
            
            # Check feasibility for next step
            # Max sum of k-1 terms starting from d+1
            # Approx (k-1)/(d+1)
            if Fraction(k-1, d+1) < new_rem:
                break
                
            # More precise check for small k
            if k > 1:
                max_sum_next = Fraction(0, 1)
                for i in range(k-1):
                    max_sum_next += Fraction(1, d + 1 + i)
                if max_sum_next < new_rem:
                    break
            
            res = dfs(new_rem, k-1, d+1, current + [d])
            if res is not None:
                return res
                
            d += 1
            
        return None

    # Try 8 terms first
    for k in range(8, 10):
        res = dfs(Fraction(1, 1), k, 5, [])
        if res is not None:
            return res
            
    raise RuntimeError("No solution found")

# EVOLVE-BLOCK-END
