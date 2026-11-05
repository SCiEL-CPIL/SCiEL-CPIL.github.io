"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    target = Fraction(5, 31)
    
    # Function to check if a solution exists with max denominator <= M and exactly k terms
    # Returns the solution list if found, else None
    def find_solution(k, M):
        # Pruning bounds
        # 1/a + ... + 1/d_k = target
        # d_1 > 1/target
        # k/d_1 >= target => d_1 <= k/target
        
        # We search for distinct denominators d1 < d2 < ... < dk <= M
        
        def dfs(start, remaining_terms, current_sum, current_denoms):
            # Time check
            if time.time() - start_time > 85:
                return None
            
            if remaining_terms == 0:
                if current_sum == target:
                    return current_denoms
                return None
            
            # Lower bound for next denominator
            # current_sum + remaining_terms / d_next = target
            # remaining_terms / d_next = target - current_sum
            # d_next = remaining_terms / (target - current_sum)
            # d_next must be >= start
            # Also d_next <= M
            
            rem = target - current_sum
            if rem <= 0:
                return None
            
            # Max possible value for next denominator to not exceed target
            # 1/d_next < rem => d_next > 1/rem
            min_d = max(start, (rem.denominator + rem.numerator - 1) // rem.numerator + 1)
            # Wait, 1/d < rem => d > 1/rem.
            # If rem = 1/10, d > 10. So min_d = 11.
            # (rem.den + rem.num - 1) // rem.num is floor(1/rem).
            # So min_d = floor(1/rem) + 1.
            
            # Min possible value for next denominator to be able to reach target
            # current_sum + 1/d_next + ... + 1/d_next (remaining_terms times) >= target
            # 1/d_next >= (target - current_sum) / remaining_terms
            # d_next <= remaining_terms / (target - current_sum)
            max_d = min(M, (remaining_terms * rem.numerator) // rem.denominator)
            
            if min_d > max_d:
                return None
            
            for d in range(min_d, max_d + 1):
                # Check distinctness: d > start ensures d > previous denominators
                # since we pick increasing denominators.
                
                new_sum = current_sum + Fraction(1, d)
                new_denoms = current_denoms + [d]
                
                res = dfs(d + 1, remaining_terms - 1, new_sum, new_denoms)
                if res:
                    return res
            
            return None
        
        return dfs(1, k, Fraction(0, 1), [])

    # Search for solutions with increasing max denominator M
    # Start from a theoretical lower bound
    # For k terms, M >= ceil(1 / (target/k)) ? No.
    # M >= ceil(k / target) is not right.
    # k/M >= target => M <= k/target. This is upper bound on M for existence?
    # No, if M is too small, we can't reach the sum.
    # We need k/M >= target => M <= k/target.
    # So if M > k/target, it's possible?
    # Actually, if M is very large, we can always pick large denominators, but sum will be small.
    # We need sum = target.
    # The minimum sum for k distinct denominators <= M is achieved by picking the largest k denominators: M, M-1, ..., M-k+1.
    # If sum(1/(M-i)) < target, then no solution exists for this M.
    
    # Let's just iterate M from a reasonable lower bound.
    # Lower bound for M:
    # The largest denominator must be at least 1/target? No.
    # For k=3, we found M=434.
    # Let's start M from 25 (theoretical min for k=4) up to 500.
    
    best_solution = None
    best_max_den = float('inf')
    
    # First, find the best 3-term solution to have a baseline
    # We can just run the search for k=3 with a high M and track the min max denominator
    # But the DFS above returns the FIRST solution found for a fixed M.
    # To find the MIN max denominator, we should iterate M upwards and return the first M that has a solution.
    
    # Since k=3 solution with M=434 exists, and k=4 might have smaller M,
    # we should check k=4, 5, 6... for small M first?
    # Or just iterate M from 25 upwards, and for each M, check k=3,4,5...
    # If any k yields a solution, return it.
    
    # Theoretical min M for k terms:
    # k/M >= 5/31 => M <= 31k/5.
    # This is an UPPER bound on M?
    # No. If M is small, the denominators are small, so reciprocals are large, so sum is large.
    # If M is large, reciprocals are small, sum is small.
    # We need sum = 5/31.
    # If we pick denominators too large (M too big), sum < 5/31.
    # If we pick denominators too small (M too small), sum > 5/31.
    # So there is a range of M.
    
    # However, we want to MINIMIZE M.
    # So we start from the smallest possible M.
    # Smallest M for k terms:
    # The largest denominator is M.
    # The smallest sum we can get with max denominator M is approx k/M.
    # If k/M < 5/31, then even with the largest possible denominators (all close to M), the sum is too small.
    # So we need k/M >= 5/31 => M <= 31k/5.
    # This means M must be LESS than or equal to 31k/5?
    # Wait. If M=100, k=4, sum ~ 4/100 = 0.04. Target 0.16. Sum too small.
    # So M must be SMALL enough to allow large enough reciprocals.
    # So M <= 31k/5 is a necessary condition for existence?
    # Yes. If M > 31k/5, then 1/M < 5/(31k).
    # Sum of k terms each <= 1/M is <= k/M < 5/31.
    # So if M > 31k/5, no solution.
    
    # Therefore, to minimize M, we should start from the LOWER bound of M?
    # No, M is the MAX denominator.
    # If M is small, denominators are small, sum is large.
    # If M is large, sum is small.
    # We want sum = 5/31.
    # So M is constrained to a range.
    # But we want to MINIMIZE M.
    # So we want the smallest M such that a solution exists.
    # But if M is too small, sum > 5/31.
    # So we must increase M until sum can be 5/31.
    # The condition "Sum <= k/M" is for the MAXIMUM possible sum?
    # No, 1/d >= 1/M. So Sum >= k/M.
    # If k/M > 5/31, it's possible (we can choose smaller denominators? No, denominators <= M).
    # Wait, denominators are DISTINCT and <= M.
    # The MAXIMUM sum is 1/1 + 1/2 + ...
    # The MINIMUM sum for max denominator M is 1/M + 1/(M-1) + ...
    
    # Let's re-evaluate.
    # We want to minimize M.
    # If M is very small, say M=10, k=4.
    # Max sum = 1/1+1/2+1/3+1/4 = 2.08 > 0.16.
    # Min sum = 1/10+1/9+1/8+1/7 = 0.42 > 0.16.
    # So for M=10, sum is always > 0.16.
    # As M increases, the min sum decreases.
    # When min sum drops below 0.16, a solution might exist.
    # So we should iterate M from small to large?
    # No, we want the SMALLEST M that works.
    # But for small M, sum is too big.
    # So we need to INCREASE M to reduce the sum.
    # So we start M from a lower bound where min_sum > target, and increase M until min_sum < target < max_sum.
    
    # Lower bound for M:
    # min_sum(M) = sum_{i=0}^{k-1} 1/(M-i)
    # We need min_sum(M) < 5/31.
    
    # Let's just iterate M from 10 upwards.
    # For each M, check if a solution exists.
    # To check existence, we can use the DFS.
    # But DFS might be slow if M is large.
    # However, we found a 3-term solution with M=434.
    # A 4-term solution might have M ~ 25-50.
    
    # Let's try M from 25 to 500.
    # For each M, try k=4, then k=3, then k=5?
    # Actually, if a solution exists for M, it exists for any M' > M (by adding terms? No, distinct denominators).
    # No, if a solution exists with max M, it doesn't imply one with max M+1.
    # But we are looking for the GLOBAL minimum max denominator.
    
    # Strategy:
    # Iterate M from 25 to 500.
    # For each M, check if there is a solution with max denominator EXACTLY M?
    # Or <= M?
    # If we find a solution with max denom <= M, we can return it?
    # No, we want the MINIMUM max denominator.
    # So if we find a solution with max denom = M, and we are iterating M upwards,
    # the first M for which we find ANY solution is the minimum max denominator.
    
    # So:
    for M in range(25, 501):
        if time.time() - start_time > 80:
            break
        
        # Try k=4 first (likely to have smaller M than k=3)
        # Then k=3
        # Then k=5
        
        for k in [4, 3, 5]:
            sol = find_solution(k, M)
            if sol:
                # Verify solution
                s = sum(Fraction(1, d) for d in sol)
                if s == target:
                    return sol

    # Fallback: Greedy or known 3-term solution
    # Known: [7, 62, 434]
    return [7, 62, 434]

# EVOLVE-BLOCK-END
