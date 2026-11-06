import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    # Try k from 8 to 256
    for k in range(8, 257):
        if time.time() > deadline:
            break
            
        # DFS to find a solution with exactly k terms
        # Target is 1/1
        # Min denominator 5
        # We store the path
        path = []
        
        def max_possible_sum(terms, min_d):
            # Sum of 1/(min_d + i) for i in 0..terms-1
            s = 0.0
            for i in range(terms):
                s += 1.0 / (min_d + i)
            return s

        def dfs(target_num, target_den, min_d, k_left):
            if time.time() > deadline:
                return False
            
            if k_left == 0:
                return target_num == target_den
            
            # Pruning: Check if max possible sum of remaining terms is >= target
            # target = target_num / target_den
            # We need target <= sum_{i=0}^{k_left-1} 1/(min_d + i)
            # Using float for pruning is risky but fast.
            # Let's use a slightly conservative float check.
            if target_num / target_den > max_possible_sum(k_left, min_d) * 1.000001:
                return False
            
            # Iterate possible denominators d
            # d must be >= min_d
            # Also 1/d <= target => d >= target_den / target_num
            # Let's find the lower bound for d
            d_min = max(min_d, math.ceil(target_den / target_num))
            
            # Upper bound for d?
            # 1/d >= target / k_left ? No.
            # The remaining sum is target - 1/d.
            # This remaining sum must be >= min possible sum of k_left-1 terms.
            # Min sum is approx 0.
            # However, we also have the constraint that we must be able to form the remainder.
            # A stronger upper bound:
            # target - 1/d <= max_possible_sum(k_left-1, d+1)
            # This implies 1/d >= target - max_possible_sum(k_left-1, d+1)
            
            # Let's just iterate d from d_min upwards.
            # How far up?
            # If d is too large, 1/d is small, and the remaining sum is close to target.
            # But the max sum of remaining terms decreases as d increases.
            
            # Heuristic upper bound:
            # If 1/d < target / k_left, then even if we pick the largest possible unit fractions for the rest, 
            # it might not work? No.
            
            # Let's iterate.
            # To avoid infinite loop or too many iterations, we can bound d.
            # If target - 1/d <= 0, break.
            # Also if target - 1/d > max_possible_sum(k_left-1, d+1), we can skip this d?
            # Actually, if target - 1/d > max_possible_sum(k_left-1, d+1), then it's impossible.
            # As d increases, 1/d decreases, so target - 1/d increases.
            # max_possible_sum(k_left-1, d+1) decreases.
            # So if the condition fails for some d, it will fail for all larger d?
            # Let L(d) = target - 1/d. L(d) increases with d.
            # Let R(d) = max_possible_sum(k_left-1, d+1). R(d) decreases with d.
            # We need L(d) <= R(d).
            # Since L increases and R decreases, there is a crossing point.
            # We can find the max d such that L(d) <= R(d).
            
            # Let's estimate the upper bound for d.
            # target - 1/d <= (k_left-1) / (d+1)  (Approximation: sum of k terms each <= 1/(d+1))
            # target <= 1/d + (k_left-1)/(d+1)
            # For large d, 1/d + (k_left-1)/(d+1) ~ k_left/d.
            # So d ~ k_left / target.
            
            d_max_est = int((k_left * target_den) / target_num) + 100
            # Cap d_max to some reasonable limit to prevent hanging if float is imprecise
            # But d is bounded by 10^7.
            
            for d in range(d_min, min(d_max_est, 10000000) + 1):
                # Check 1/d <= target
                if d * target_num < target_den:
                    break
                
                # Check if remainder can be formed
                rem_num = target_num * d - target_den
                rem_den = target_den * d
                if rem_num < 0:
                    break
                if rem_num == 0:
                    if k_left == 1:
                        path.append(d)
                        return True
                    else:
                        # Cannot form 0 with k_left-1 > 0 terms
                        continue
                
                # Simplify remainder
                g = math.gcd(rem_num, rem_den)
                rn = rem_num // g
                rd = rem_den // g
                
                # Pruning: Check max sum for remainder
                # rem = rn/rd
                # Need rn/rd <= max_possible_sum(k_left-1, d+1)
                if rn / rd > max_possible_sum(k_left - 1, d + 1) * 1.000001:
                    continue
                
                path.append(d)
                if dfs(rn, rd, d + 1, k_left - 1):
                    return True
                path.pop()
                
            return False

        if dfs(1, 1, 5, k):
            return path

    # If no solution found with k up to 256, return a fallback?
    # The task guarantees a solution exists in 1..256.
    # If we fail, we return an empty list or raise error?
    # "invalid result ... scores 0".
    # We should ensure we find one.
    # Given the constraints, k=8 should work.
    raise RuntimeError("No solution found")
