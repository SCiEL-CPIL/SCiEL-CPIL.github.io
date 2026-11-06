"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    # Minimum is 10 terms (proven: max 9-term sum 1/5+...+1/13 < 1)
    # Find a 10-term solution using integer arithmetic with a common denominator approach.
    # Strategy: pick a target T, find 10 distinct divisors d_i of T in [5, 10^7]
    # such that sum(T/d_i) = T. Equivalently, sum of 10 distinct values w_i = T/d_i
    # (where w_i divides T) sums to T.

    # Use T = LCM of a set of small numbers to keep divisors manageable.
    # T = 2520 = 2^3 * 3^2 * 5 * 7 has many divisors.
    # But we need sum of 10 distinct T/d_i = T, i.e., sum of 10 distinct w_i = T
    # where each w_i = T/d_i for some integer d_i in [5, 10^7].
    # d_i in [5, 10^7] means w_i = T/d_i, so w_i must be a divisor of T and T/w_i in [5, 10^7].
    # For T=2520: w_i divides 2520, and 2520/w_i in [5, 10^7] => w_i in [1, 504].
    # So we need 10 distinct divisors w_i of 2520, each in [1, 504], summing to 2520.
    # The divisors of 2520 in [1, 504] are all divisors of 2520 (since 2520/5=504).
    # Divisors of 2520: 1,2,3,4,5,6,7,8,9,10,12,14,15,18,20,21,24,28,30,35,36,40,42,45,56,60,63,70,72,84,90,105,120,126,140,168,180,210,252,280,315,360,420,504,630,840,1260,2520
    # Those <= 504: 1,2,3,4,5,6,7,8,9,10,12,14,15,18,20,21,24,28,30,35,36,40,42,45,56,60,63,70,72,84,90,105,120,126,140,168,180,210,252,280,315,360,420,504

    # We need 10 distinct values from this set summing to 2520.
    # Let's do a DP/subset-sum to find 10 distinct divisors summing to 2520.

    T = 2520
    # Get all valid w values (divisors of T with T/w in [5, 10^7])
    divisors = []
    for w in range(1, T + 1):
        if T % w == 0:
            d = T // w
            if 5 <= d <= 10000000:
                divisors.append(w)
    
    # DP: can we pick exactly 10 distinct values from divisors summing to T?
    # State: (sum, count, last_index) -> feasible
    # Use a set of achievable (sum, count) pairs with the actual values.
    
    # More efficient: DP over the sorted divisors, tracking achievable (sum, count)
    # with reconstruction.
    divisors.sort()
    n_div = len(divisors)
    
    # dp[count][sum] = list of indices used, or None
    # To save memory, use a dict: (count, sum) -> tuple of chosen w values
    from collections import defaultdict
    dp = defaultdict(set)  # (count, sum) -> set of frozensets of indices (too heavy)
    
    # Better: use a different approach. Since n_div is ~45, and we need 10 from 45,
    # use backtracking with pruning.
    
    def find_10_divisors(target_sum, divs, k, start_idx, current_sum, current_list):
        if time.time() > deadline - 3.0:
            return None
        if k == 0:
            if current_sum == target_sum:
                return list(current_list)
            return None
        if len(divs) - start_idx < k:
            return None
        # Upper bound: max possible sum with k remaining
        max_possible = sum(divs[-k:]) if k <= len(divs) else sum(divs)
        if current_sum + max_possible < target_sum:
            return None
        # Lower bound: min possible sum with k remaining
        min_possible = sum(divs[start_idx:start_idx + k]) if start_idx + k <= len(divs) else sum(divs[start_idx:])
        if current_sum + min_possible > target_sum:
            return None
        
        remaining_needed = target_sum - current_sum
        for i in range(start_idx, len(divs)):
            w = divs[i]
            if current_sum + w > target_sum:
                break
            current_list.append(w)
            result = find_10_divisors(target_sum, divs, k - 1, i + 1, current_sum + w, current_list)
            if result is not None:
                return result
            current_list.pop()
        return None
    
    result_w = find_10_divisors(T, divisors, 10, 0, 0, [])
    if result_w is not None:
        # Convert back to denominators
        solution = [T // w for w in result_w]
        # Verify
        if len(set(solution)) == 10 and all(5 <= d <= 10000000 for d in solution):
            s = sum(1/d for d in solution)
            # Verify exactly using integer arithmetic
            from fractions import Fraction
            exact_s = sum(Fraction(1, d) for d in solution)
            if exact_s == 1:
                return sorted(solution)
    
    # Fallback: use a larger target T
    # T = 27720 = LCM(1..15) approximately. Let's use T = 360360 = LCM(5..14)
    # Actually, let's try T = 2520 * k for various k, or use a different strategy.
    
    # Alternative: direct search with Fraction but much faster pruning
    from fractions import Fraction
    
    def fast_dfs(remaining_num, remaining_den, k, min_denom):
        """Find k distinct denominators >= min_denom summing to remaining_num/remaining_den."""
        if time.time() > deadline - 2.0:
            return None
        if k == 0:
            if remaining_num == 0:
                return []
            return None
        if remaining_num <= 0:
            return None
        
        # Upper bound: max sum with k terms starting from min_denom
        # Use float for speed
        ub = sum(1.0 / (min_denom + i) for i in range(k))
        if ub < remaining_num / remaining_den - 1e-12:
            return None
        
        # Max denominator to try: 1/d >= remaining/k => d <= k*remaining_den/remaining_num
        max_d = int(k * remaining_den / remaining_num) + 1
        max_d = min(max_d, 10000000)
        
        for d in range(min_denom, max_d + 1):
            # new_remaining = remaining - 1/d = (remaining_num*d - remaining_den) / (remaining_den*d)
            new_num = remaining_num * d - remaining_den
            if new_num < 0:
                break
            new_den = remaining_den * d
            # Reduce fraction
            g = gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g
            sub = fast_dfs(new_num, new_den, k - 1, d + 1)
            if sub is not None:
                return [d] + sub
        return None
    
    sol = fast_dfs(1, 1, 10, 5)
    if sol is not None:
        from fractions import Fraction
        exact_s = sum(Fraction(1, d) for d in sol)
        if exact_s == 1 and len(set(sol)) == 10 and all(5 <= d <= 10000000 for d in sol):
            return sorted(sol)
    
    # Final fallback: DP on T=2520 for any number of terms
    T = 2520
    divisors = []
    for w in range(1, T + 1):
        if T % w == 0:
            d = T // w
            if 5 <= d <= 10000000:
                divisors.append(w)
    divisors.sort()
    
    # Find minimum number of distinct divisors summing to T
    # BFS/DP: dp[sum] = min count to achieve sum, with reconstruction
    INF = float('inf')
    dp_count = [INF] * (T + 1)
    dp_prev = [None] * (T + 1)  # (prev_sum, w)
    dp_count[0] = 0
    
    for w in divisors:
        for s in range(T, w - 1, -1):
            if dp_count[s - w] + 1 < dp_count[s]:
                dp_count[s] = dp_count[s - w] + 1
                dp_prev[s] = (s - w, w)
    
    if dp_count[T] < INF:
        # Reconstruct
        solution_w = []
        s = T
        while s > 0:
            prev_s, w = dp_prev[s]
            solution_w.append(w)
            s = prev_s
        solution = [T // w for w in solution_w]
        if len(set(solution)) == len(solution) and all(5 <= d <= 10000000 for d in solution):
            from fractions import Fraction
            exact_s = sum(Fraction(1, d) for d in solution)
            if exact_s == 1:
                return sorted(solution)
    
    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
