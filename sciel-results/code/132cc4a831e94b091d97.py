import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85

    def max_sum(m, start_d):
        """Calculate sum of 1/d for d from start_d to start_d+m-1"""
        s = 0.0
        for i in range(m):
            s += 1.0 / (start_d + i)
        return s

    def dfs(rem_sum, min_d, terms, k_left):
        if time.time() > deadline:
            return None

        if k_left == 0:
            if rem_sum == 0:
                return terms
            return None

        # Pruning: Check if max possible sum is sufficient
        # Max sum using k_left smallest available denominators
        # Note: rem_sum is Fraction, max_sum is float.
        # We need rem_sum <= max_sum
        if rem_sum > max_sum(k_left, min_d + 1) + 1e-9:
            return None

        if k_left == 1:
            # rem_sum must be 1/d
            d = rem_sum.denominator // rem_sum.numerator
            if rem_sum.numerator * d == rem_sum.denominator and d > min_d and d <= 10000000:
                return terms + [d]
            return None

        # Lower bound for next d
        # 1/d <= rem_sum => d >= 1/rem_sum
        lower_d = max(min_d + 1, (rem_sum.denominator + rem_sum.numerator - 1) // rem_sum.numerator)
        
        # Upper bound for next d
        # We need rem_sum - 1/d <= max_sum(k_left-1, d+1)
        # Let's find the largest d such that this holds.
        # Since max_sum decreases as d increases, we can binary search or just iterate from lower_d upwards and break when condition fails?
        # Actually, as d increases, 1/d decreases, so rem_sum - 1/d increases.
        # And max_sum(k_left-1, d+1) decreases.
        # So the LHS increases and RHS decreases.
        # There is a point where LHS > RHS, and for all larger d, it will hold.
        # So we iterate d from lower_d and break when LHS > RHS.
        
        for d in range(lower_d, 10000001):
            # Optimization: if d is too large, 1/d is small, rem_sum - 1/d is large, max_sum is small.
            # Check condition
            next_rem = rem_sum - Fraction(1, d)
            if next_rem < 0:
                break
                
            # Check if next_rem can be formed by k_left-1 terms starting from d+1
            if next_rem > max_sum(k_left - 1, d + 1) + 1e-9:
                break

            # Also, we need next_rem > 0 (handled)
            # And we need to ensure we can reach exactly next_rem.
            # The DFS will handle that.
            
            res = dfs(next_rem, d, terms + [d], k_left - 1)
            if res is not None:
                return res
                
        return None

    for k in range(8, 20):
        res = dfs(Fraction(1, 1), 4, [], k)
        if res is not None:
            return res
            
    # Fallback
    return []
# EVOLVE-BLOCK-END
