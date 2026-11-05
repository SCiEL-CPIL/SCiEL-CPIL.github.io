import time
import sys
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s limit (or 90s if that's the hard limit, but 60s is standard for these tasks usually, let's use 55s to be safe)
    
    def elapsed():
        return time.time() - start_time
    
    def time_ok():
        return elapsed() < deadline

    def find_solution(k):
        """
        Finds k distinct integers d_1 < ... < d_k >= 5 such that sum(1/d_i) = 1.
        Returns the list of denominators or None if not found.
        """
        # Pre-calculate some bounds?
        # We use DFS with pruning.
        
        def dfs(rem, min_d, depth, max_depth, path):
            if not time_ok():
                return None
            if rem == 0:
                return path
            if depth == max_depth:
                return None
            
            terms_left = max_depth - depth
            
            # Pruning 1: Upper bound on the sum of remaining terms
            # The maximum sum is achieved by the smallest possible denominators: min_d, min_d+1, ...
            max_sum = sum(Fraction(1, min_d + i) for i in range(terms_left))
            if rem > max_sum:
                return None
            
            # Pruning 2: Lower bound on the sum of remaining terms
            # The minimum sum is achieved by the largest possible denominators.
            # Denominators are <= 10^7.
            # So min_sum = terms_left / 10^7.
            if rem < Fraction(terms_left, 10**7):
                return None
            
            # Determine the range for the next denominator d
            # 1/d <= rem  => d >= 1/rem
            d_min = max(min_d, (1 + rem.denominator - 1) // rem.numerator)
            
            # Upper bound for d:
            # We need 1/d + sum_{j=1}^{terms_left-1} 1/(d+j) >= rem
            # A loose upper bound is d <= terms_left / rem
            d_max = (terms_left * rem.denominator) // rem.numerator
            
            if d_min > d_max:
                return None
            
            # Iterate through possible denominators
            # To find a solution quickly, we might want to try specific orders.
            # Smaller d leaves larger remainder, which might be harder to fill with fewer terms?
            # Actually, larger d leaves smaller remainder.
            # Let's just iterate from d_min to d_max.
            
            for d in range(d_min, d_max + 1):
                if not time_ok():
                    return None
                new_rem = rem - Fraction(1, d)
                if new_rem < 0:
                    break
                res = dfs(new_rem, d + 1, depth + 1, max_depth, path + [d])
                if res is not None:
                    return res
            return None

        return dfs(Fraction(1, 1), 5, 0, k, [])

    # The minimum possible k is 10.
    # We try k=10 first. If found, it's the optimal solution.
    for k in range(10, 21):
        if not time_ok():
            break
        res = find_solution(k)
        if res is not None:
            return res
    
    # If we still haven't found a solution, which should not happen for k=10,
    # we raise an error or return a dummy. But the task requires a valid return.
    # However, based on the analysis, k=10 is possible.
    raise RuntimeError("No solution found within time limit")
