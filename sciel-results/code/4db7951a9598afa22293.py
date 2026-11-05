"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Try to find a solution with minimal number of terms k
    # We know k=2,3,4,5,6,7,8,9,10 are candidates
    # The seed program found k=10. We try to find k < 10.
    
    def search_k(k, min_den=5):
        # Backtracking search for k distinct denominators >= min_den summing to 1
        # To ensure we find the "smallest" or just any, we iterate
        # Pruning:
        # 1. Sum of remaining reciprocals must be achievable.
        # 2. Max sum of (k-i) terms starting from d+1 is 1/(d+1) + ...
        #    If current_sum + max_possible < 1, prune? No, if max_possible < 1-current_sum, prune.
        # 3. Min sum of (k-i) terms starting from d+1 is (k-i) / (some large number).
        #    If current_sum + min_possible > 1, prune.
        
        current_sum = Fraction(0, 1)
        denominators = []
        
        def backtrack(idx, current_sum, last_den):
            if time.time() > deadline:
                return None
            
            if idx == k:
                if current_sum == 1:
                    return denominators[:]
                return None
            
            # Remaining terms: k - idx
            rem = k - idx
            
            # We need to choose 'rem' denominators d_i > last_den
            # such that sum(1/d_i) = 1 - current_sum
            
            target_rem = 1 - current_sum
            if target_rem <= 0:
                return None
            
            # Max possible sum for 'rem' terms starting from last_den + 1
            # is 1/(last_den+1) + 1/(last_den+2) + ...
            # Approximation: rem / (last_den + 1) is an upper bound on sum? No, sum is less.
            # Actually, sum_{j=1}^rem 1/(last_den + j) < rem / (last_den + 1)
            # If target_rem > rem / (last_den + 1), it's impossible.
            if target_rem > rem / (last_den + 1):
                return None
            
            # Min possible sum for 'rem' terms starting from last_den + 1
            # is when denominators are as large as possible.
            # But we don't have an upper bound on denominators, only 10000000.
            # However, if target_rem is very small, we might need large denominators.
            # But we are minimizing count, so we expect denominators to be somewhat small.
            
            # Lower bound for next denominator d
            # 1/d <= target_rem / rem  => d >= rem / target_rem
            # Also d > last_den
            min_d = max(last_den + 1, int(rem / target_rem) + 1)
            
            # Upper bound for next denominator d
            # 1/d >= target_rem / rem is not required, but 1/d must be large enough
            # to allow the remaining rem-1 terms to sum to target_rem - 1/d.
            # Max sum of rem-1 terms is (rem-1) / (d+1).
            # So target_rem - 1/d <= (rem-1) / (d+1)
            # This gives a bound on d.
            # target_rem <= 1/d + (rem-1)/(d+1)
            
            # Let's iterate d from min_d upwards
            # We can limit the search range for d.
            # If 1/d > target_rem, break (since 1/d decreases)
            # Actually, we need 1/d < target_rem (strictly, since other terms positive)
            # So d > 1/target_rem
            max_d = int(1 / target_rem) - 1
            if max_d < min_d:
                return None
            
            # Also, we must be able to fit 'rem-1' terms after d.
            # The smallest possible sum for rem-1 terms > d is (rem-1) / (d + rem - 1) ? No.
            # The largest denominators we can pick are unbounded, but sum must be exact.
            
            # Optimization: If rem == 1, then 1/d must equal target_rem.
            if rem == 1:
                if target_rem.denominator == 1:
                    d = target_rem.numerator
                    if d > last_den and d <= 10000000:
                        denominators.append(d)
                        return denominators[:]
                return None
            
            # Iterate d
            # To avoid huge loops, we can stop if 1/d becomes too small
            # such that even with infinite denominators we can't reach target_rem? No.
            # We just iterate.
            
            # Heuristic: try smaller d first? Or larger?
            # Smaller d contributes more, so it's more likely to satisfy the sum.
            
            for d in range(min_d, max_d + 1):
                if time.time() > deadline:
                    return None
                
                next_sum = current_sum + Fraction(1, d)
                if next_sum > 1:
                    break # Since d is increasing, 1/d is decreasing, so next_sum will decrease?
                    # Wait. d increases -> 1/d decreases -> next_sum decreases.
                    # So if next_sum > 1, we can continue? No.
                    # If d is small, 1/d is large.
                    # We start with min_d.
                    # If 1/min_d is too large, next_sum > 1.
                    # As d increases, 1/d decreases, so next_sum decreases.
                    # So we should continue until next_sum <= 1.
                    # But we already ensured d > 1/target_rem, so 1/d < target_rem.
                    # So next_sum = current_sum + 1/d < current_sum + target_rem = 1.
                    # So next_sum < 1 is always true in the loop range.
                    
                denominators.append(d)
                res = backtrack(idx + 1, next_sum, d)
                if res:
                    return res
                denominators.pop()
                
                # Pruning: if current_sum + 1/d + (rem-1)/(d+1) < 1, 
                # then even if we pick the smallest possible subsequent denominators (d+1, d+2...),
                # the sum will be too small?
                # Max sum of rem-1 terms starting from d+1 is sum_{j=1}^{rem-1} 1/(d+j).
                # If current_sum + 1/d + max_possible < 1, then we need larger 1/d?
                # No, 1/d is fixed for this step.
                # If max possible sum of remaining is less than needed, we fail.
                # Needed = 1 - (current_sum + 1/d).
                # Max remaining = sum_{j=1}^{rem-1} 1/(d+j).
                # If Max remaining < Needed, then we must try a smaller d (which gives larger 1/d, reducing Needed, but also reducing Max remaining?).
                # Actually, if we increase d, 1/d decreases, Needed increases.
                # And Max remaining decreases (denominators larger).
                # So if it fails for d, it will likely fail for larger d.
                # Let's check:
                # Needed(d) = 1 - S - 1/d. As d increases, Needed increases.
                # MaxRem(d) = sum 1/(d+j). As d increases, MaxRem decreases.
                # So if MaxRem(d) < Needed(d), then for d' > d, MaxRem(d') < MaxRem(d) < Needed(d) < Needed(d').
                # So we can break!
                
                max_rem = sum(Fraction(1, d + j) for j in range(1, rem))
                needed = 1 - next_sum
                if max_rem < needed:
                    break

            return None

        return backtrack(0, Fraction(0, 1), 4)

    # Try k from 2 to 10
    # We know seed found 10. So we try 2..9.
    # If we find one, return it.
    # If not, return the seed result (or re-run seed logic if time permits, but seed logic is fast).
    
    # Actually, let's just run the seed logic first to have a fallback?
    # No, seed logic is O(2520^2) roughly, which is fast.
    # But we want to improve score.
    
    best_result = None
    
    # Try small k
    for k in range(2, 10):
        if time.time() > deadline - 5:
            break
        res = search_k(k)
        if res:
            best_result = res
            break # Found minimal k
    
    if best_result:
        return best_result
    
    # Fallback to seed program logic if no better found
    # Seed program:
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    
    raise RuntimeError("no baseline representation")

# EVOLVE-BLOCK-END
