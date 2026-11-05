"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction
from functools import lru_cache
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    # Strategy: Find minimal Egyptian fraction with denominators >= 5.
    # Key insight: To minimize term count, use smallest denominators.
    # Lower bound: sum of 1/5 + 1/6 + ... + 1/(5+k-1) must be >= 1.
    # For k=10: 1/5+1/6+1/7+1/8+1/9+1/10+1/11+1/12+1/13+1/14 ≈ 0.2+0.167+0.143+0.125+0.111+0.1+0.091+0.083+0.077+0.071 = 1.168 > 1
    # So 10 terms is feasible. Can we do fewer?
    # For k=9: 1/5+...+1/13 ≈ 1.097 > 1, so 9 might be possible.
    # For k=8: 1/5+...+1/12 ≈ 1.020 > 1, so 8 might be possible.
    # For k=7: 1/5+...+1/11 ≈ 0.942 < 1, so 7 is impossible.
    # So minimum is 8 or 9 or 10 terms.
    
    # Use DFS with pruning: try denominators in increasing order, 
    # but allow skipping some to find non-consecutive solutions.
    
    # Precompute: for a given minimum denominator, what's the max sum 
    # achievable with 'remaining_terms' more terms?
    def max_sum_from(min_denom, remaining_terms):
        """Maximum sum of 'remaining_terms' distinct unit fractions with denominators >= min_denom."""
        if remaining_terms <= 0:
            return Fraction(0)
        # Greedy upper bound: use the smallest possible denominators
        s = Fraction(0)
        d = min_denom
        for _ in range(remaining_terms):
            s += Fraction(1, d)
            d += 1
        return s
    
    def min_sum_from(min_denom, remaining_terms):
        """Minimum sum of 'remaining_terms' distinct unit fractions with denominators >= min_denom."""
        if remaining_terms <= 0:
            return Fraction(0)
        # Use the largest possible denominators (but they're unbounded, so min is 0)
        # Actually for pruning, we need: can we still reach target?
        # Min sum approaches 0 as denominators grow, so this isn't useful for lower bound.
        # Instead, use: the minimum additional sum we can add is > 0, but we need 
        # to check if remaining_target <= max_sum_from(...)
        return Fraction(0)
    
    # DFS to find a solution with exactly k terms
    def solve_k(k, max_denom=10000000):
        """Find k distinct denominators >= 5 summing to 1."""
        if k < 1 or k > 256:
            return None
        
        # Try DFS with pruning
        def dfs(start_denom, remaining_terms, remaining_sum, current):
            if time.time() > deadline:
                return None
            if remaining_terms == 0:
                if remaining_sum == 0:
                    return current
                return None
            if remaining_terms == 1:
                # Need 1/d = remaining_sum, so d = 1/remaining_sum
                if remaining_sum > 0:
                    d = 1 / remaining_sum
                    if d == int(d) and int(d) >= start_denom and int(d) <= max_denom:
                        return current + [int(d)]
                return None
            
            # Pruning: can we achieve remaining_sum with remaining_terms fractions?
            # Max possible sum from start_denom with remaining_terms terms:
            max_possible = max_sum_from(start_denom, remaining_terms)
            if remaining_sum > max_possible:
                return None
            
            # Min possible sum is > 0, but we need remaining_sum >= 0
            if remaining_sum < 0:
                return None
            
            # Try each possible next denominator
            # Upper bound on next denominator: 1/d <= remaining_sum, so d >= ceil(1/remaining_sum)
            # Also d >= start_denom
            if remaining_sum > 0:
                min_d = max(start_denom, math.ceil(1 / remaining_sum))
            else:
                min_d = start_denom
            
            # Upper bound: we need to leave room for remaining_terms-1 more terms
            # The smallest sum we can add with remaining_terms-1 terms starting from d+1 is > 0
            # But we need remaining_sum - 1/d >= 0, so d >= ceil(1/remaining_sum)
            # Also, the remaining sum after choosing d must be achievable:
            # remaining_sum - 1/d <= max_sum_from(d+1, remaining_terms-1)
            
            # Limit search range for efficiency
            # Try denominators from min_d, but limit how far we go
            # If remaining_sum is small, d must be large
            max_d_to_try = min(max_denom, min_d + 100000)  # Limit search space
            
            for d in range(min_d, max_d_to_try + 1):
                if time.time() > deadline:
                    return None
                
                # Check if choosing d leaves a solvable subproblem
                new_remaining = remaining_sum - Fraction(1, d)
                if new_remaining < 0:
                    break  # d too small, 1/d > remaining_sum
                
                # Check if new_remaining is achievable with remaining_terms-1 terms from d+1
                max_next = max_sum_from(d + 1, remaining_terms - 1)
                if new_remaining > max_next:
                    continue  # Even with smallest denominators, can't reach new_remaining
                
                # Also check: new_remaining must be >= 0
                if new_remaining < 0:
                    continue
                
                res = dfs(d + 1, remaining_terms - 1, new_remaining, current + [d])
                if res is not None:
                    return res
            
            return None
        
        return dfs(5, k, Fraction(1), [])
    
    # Try k = 8, 9, 10, ... until we find a solution
    best_result = None
    for k in range(8, 16):
        if time.time() > deadline - 5:
            break
        result = solve_k(k)
        if result is not None:
            # Verify
            if len(result) == k and all(5 <= d <= 10000000 for d in result):
                if len(set(result)) == len(result):
                    s = sum(Fraction(1, d) for d in result)
                    if s == 1:
                        best_result = result
                        break
    
    if best_result:
        return best_result
    
    # Fallback: use a known valid solution if DFS fails
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/21 + 1/30 + 1/42 + 1/70 + 1/140 + 1/210 + 1/420 + 1/840 + 1/1260 + 1/2520
    # Let me verify this sums to 1... actually let me just use a simpler verified one.
    
    # Use: 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30
    # This sums to more than 1, so not valid.
    
    # Just return empty if we can't find one (shouldn't happen)
    return []
# EVOLVE-BLOCK-END
