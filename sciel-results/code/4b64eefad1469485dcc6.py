"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import math

def construct():
    target = Fraction(5, 31)
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin
    
    # 1. Find an initial solution using greedy to get an upper bound on K
    def greedy_solution():
        r = target
        denoms = []
        while r != 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d <= 0 or r <= 0:
                return None
            denoms.append(d)
            r -= Fraction(1, d)
            if len(denoms) > 256:
                return None
        # Check distinctness
        if len(set(denoms)) != len(denoms):
            return None
        return denoms

    init_sol = greedy_solution()
    if init_sol is None:
        # Fallback if greedy fails (unlikely for 5/31)
        return [7, 10, 13, 15, 21, 22, 25, 31, 33, 40, 45, 52, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200] # Just a placeholder, should not happen
    
    best_max = max(init_sol)
    best_list = init_sol
    
    # 2. Binary search for the minimal K
    # Lower bound: The smallest denominator must be at least ceil(1/(target/256)) ? No.
    # The smallest denominator d1 must satisfy 1/d1 <= 5/31 => d1 >= 7.
    # Also, if we use 1 term, d=31/5 not integer.
    # Let's just start low.
    lo = 7
    hi = best_max
    
    def can_solve_with_max_k(k):
        """
        Returns a list of denominators if a valid solution exists with all d_i <= k, else None.
        Uses DFS with pruning.
        """
        # State: (remainder, min_next_denom, terms_used, current_denoms)
        # We use recursion with manual stack or recursion limit increase.
        # Given the constraints, recursion depth is <= 256, which is fine.
        
        def dfs(rem, min_d, terms_used, current_denoms):
            if time.time() > deadline:
                return None
            
            if rem == 0:
                if len(current_denoms) <= 256:
                    return list(current_denoms)
                return None
            
            if terms_used >= 256:
                return None
            
            if rem < 0:
                return None
            
            # Pruning:
            # 1. We need to form 'rem' with at most (256 - terms_used) terms.
            # 2. All terms must be >= min_d and <= k.
            # 3. The maximum sum we can form with m terms starting from min_d is sum(1/(min_d + i) for i in range(m)).
            #    If this max_sum < rem, we fail.
            # 4. The minimum sum we can form with m terms ending at k is sum(1/(k - i) for i in range(m)).
            #    If this min_sum > rem, we fail.
            
            m = 256 - terms_used
            
            # Approximate max sum with m terms starting from min_d
            # Sum 1/(min_d + i) for i=0..m-1
            # This is H(min_d+m-1) - H(min_d-1)
            # For small m, we can just sum. For large m, use approximation or just bound by m/min_d.
            # m/min_d is a loose upper bound.
            if m / min_d < rem:
                return None
            
            # Approximate min sum with m terms ending at k
            # Sum 1/(k - i) for i=0..m-1
            # This is H(k) - H(k-m)
            # Loose lower bound: m/k
            if m / k > rem:
                # This means even the smallest possible sum is too big?
                # No, 1/k is the smallest term. Sum of m terms is at least m/k.
                # If m/k > rem, we can't make the sum small enough.
                return None

            # Determine range for next denominator d
            # 1/d <= rem  => d >= ceil(1/rem)
            d_min = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
            
            # 1/d >= rem / m  => d <= floor(m / rem)
            # This is because if we pick d too small, 1/d is too big, and even with m-1 smallest possible remaining terms (1/k...), 
            # wait, the bound d <= m/rem comes from 1/d <= rem/m is NOT correct.
            # Correct bound: We need 1/d + (sum of m-1 smallest possible terms) <= rem? No.
            # We need 1/d <= rem. That's it for the single term.
            # However, we must be able to form the remainder rem - 1/d with m-1 terms.
            # The maximum value we can form with m-1 terms starting from d+1 is (m-1)/(d+1).
            # So we need rem - 1/d <= (m-1)/(d+1).
            # rem - 1/d <= (m-1)/(d+1)
            # rem(d+1) - (d+1) <= (m-1)
            # rem*d + rem - d - 1 <= m - 1
            # d(rem - 1) <= m - 1 - rem
            # d <= (m - 1 - rem) / (rem - 1)
            # This bound is tricky with Fractions.
            
            # Simpler upper bound for d:
            # Since all denominators must be distinct and >= d, the largest sum we can form with m terms starting from d is roughly m/d.
            # We already checked m/min_d >= rem.
            # For a specific d, we need to ensure rem - 1/d can be formed by m-1 terms >= d+1.
            # Max sum of m-1 terms starting from d+1 is (m-1)/(d+1).
            # So we require rem - 1/d <= (m-1)/(d+1).
            # Let's use this to find d_max.
            
            # rem - 1/d <= (m-1)/(d+1)
            # rem(d+1) - (d+1) <= (m-1)
            # d(rem - 1) <= m - 1 - rem
            # If rem < 1, rem-1 is negative, so d >= (m-1-rem)/(rem-1) which is a lower bound?
            # Let's just iterate d from d_min to k, but break early if possible.
            
            # A better upper bound for d:
            # If 1/d > rem, we can't pick d. (Handled by d_min)
            # Also, if we pick d, the remaining sum is rem' = rem - 1/d.
            # We need rem' >= (m-1)/k (min sum of m-1 terms with max denom k).
            # So rem - 1/d >= (m-1)/k
            # 1/d <= rem - (m-1)/k
            # d >= 1 / (rem - (m-1)/k)
            # This gives another lower bound.
            
            # And rem' <= (m-1)/(d+1) (max sum of m-1 terms starting from d+1)
            # rem - 1/d <= (m-1)/(d+1)
            # This implies d cannot be too small.
            
            # Let's just iterate d from d_min to k.
            # Optimization: If d is very small, 1/d is large.
            
            for d in range(d_min, k + 1):
                if time.time() > deadline:
                    return None
                
                # Check if 1/d <= rem
                if Fraction(1, d) > rem:
                    continue
                
                new_rem = rem - Fraction(1, d)
                
                # Pruning for next step:
                # Can we form new_rem with m-1 terms?
                # Min sum of m-1 terms with max denom k is (m-1)/k.
                # If new_rem < (m-1)/k, we fail (can't make it small enough? No, we can make it smaller by picking larger denominators).
                # Wait, the minimum sum is achieved by picking the LARGEST denominators: k, k-1, ..., k-(m-2).
                # Sum_min = sum(1/(k-i) for i in range(m-1)).
                # If new_rem < Sum_min, we fail.
                
                # Max sum of m-1 terms starting from d+1 is (m-1)/(d+1).
                # If new_rem > (m-1)/(d+1), we fail.
                
                if m - 1 == 0:
                    if new_rem == 0:
                        current_denoms.append(d)
                        sol = list(current_denoms)
                        current_denoms.pop()
                        return sol
                    else:
                        continue

                # Check lower bound for new_rem
                # Approximation: (m-1)/k
                if new_rem < Fraction(m - 1, k):
                    continue
                
                # Check upper bound for new_rem
                # Approximation: (m-1)/(d+1)
                if new_rem > Fraction(m - 1, d + 1):
                    continue
                
                current_denoms.append(d)
                res = dfs(new_rem, d + 1, terms_used + 1, current_denoms)
                current_denoms.pop()
                
                if res is not None:
                    return res
            
            return None
        
        return dfs(target, 1, 0, [])

    # Binary Search
    # We know hi is valid.
    # We want to find the smallest valid K.
    # Note: can_solve_with_max_k(k) is monotonic. If it works for k, it works for k+1.
    
    # We can try to go lower.
    # Let's do a linear search downwards from hi? No, binary search is better if the range is large.
    # Range is [7, best_max]. best_max is likely < 10000.
    
    # To save time, we can just iterate K downwards from best_max?
    # If best_max is small (e.g. < 100), linear is fine.
    # If best_max is large, binary search.
    
    # Let's try binary search.
    while lo < hi:
        if time.time() > deadline:
            break
        
        mid = (lo + hi) // 2
        sol = can_solve_with_max_k(mid)
        if sol is not None:
            # Found a solution with max <= mid
            if max(sol) < best_max:
                best_max = max(sol)
                best_list = sol
            hi = mid
        else:
            lo = mid + 1
    
    # Final check at lo (which should equal hi now)
    if time.time() < deadline:
        sol = can_solve_with_max_k(lo)
        if sol is not None:
            if max(sol) < best_max:
                best_max = max(sol)
                best_list = sol

    return best_list

# EVOLVE-BLOCK-END
