from fractions import Fraction
import math

def construct():
    # Target: sum of 1/n_i = 1, n_i distinct, 5 <= n_i <= 10^7
    # Minimize count n.
    # Known minimal count for denom >= 5 is 5.
    # We will search for n=5, 6, ...
    
    target = Fraction(1, 1)
    
    # Helper to check if a solution exists for a given n
    def solve(n):
        # Backtracking search
        # State: remaining sum, min allowed denom, terms left
        # Returns list of denominators or None
        
        def backtrack(remaining, min_d, terms_left, chosen):
            if terms_left == 0:
                return chosen if remaining == 0 else None
            
            if remaining <= 0:
                return None
            
            # Pruning 1: Max possible sum with 'terms_left' terms starting from min_d
            # Max sum is sum(1/i for i in range(min_d, min_d + terms_left))
            max_sum = 0
            for i in range(terms_left):
                max_sum += Fraction(1, min_d + i)
            if max_sum < remaining:
                return None
            
            # Pruning 2: Min possible sum?
            # The minimum sum is achieved by taking the largest possible denominators.
            # But we don't have an upper bound on denominators other than 10^7.
            # However, if remaining is very small, we might need large denominators.
            # But we are minimizing count, so n is small.
            
            # Determine range for next denominator d
            # 1/d <= remaining  => d >= 1/remaining
            min_d_from_sum = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
            start_d = max(min_d, min_d_from_sum)
            
            # Upper bound for d:
            # We need remaining - 1/d to be achievable with terms_left - 1 terms > d.
            # Max sum of terms_left - 1 terms > d is sum(1/i for i in range(d+1, d + terms_left))
            # So we need remaining - 1/d <= max_sum_remaining
            # 1/d >= remaining - max_sum_remaining
            # If remaining - max_sum_remaining <= 0, then any d satisfying 1/d <= remaining is ok for this bound?
            # No, if remaining - max_sum_remaining is negative, it means even the max sum of remaining terms is larger than needed, which is always true if we can go to infinity?
            # Wait. The sum of remaining terms must be EXACTLY remaining - 1/d.
            # The MAX possible sum of the remaining terms is S_max = sum(1/i for i in range(d+1, d+terms_left)).
            # If remaining - 1/d > S_max, it's impossible.
            # So 1/d >= remaining - S_max.
            # This gives an upper bound on d if remaining - S_max > 0.
            # If remaining - S_max <= 0, then this constraint doesn't restrict d from above (other than 1/d <= remaining).
            
            # Let's iterate d from start_d upwards.
            # We can stop if 1/d < remaining - S_max? No.
            # Let's just iterate and prune inside.
            
            # To limit iterations, note that d cannot be too large.
            # If d is large, 1/d is small.
            # Let's set a reasonable cap or break condition.
            # If terms_left == 1:
            #   remaining must be 1/d exactly.
            #   So d = 1/remaining. Must be integer.
            
            if terms_left == 1:
                if remaining.denominator == 1:
                    d = remaining.numerator
                    if d >= start_d and d <= 10000000:
                        return chosen + [d]
                return None
            
            # Iterate d
            # We can bound d.
            # d must be such that remaining - 1/d >= min_possible_sum(terms_left-1, d+1)
            # Min possible sum is not bounded from below easily (approaches 0).
            # But max possible sum of remaining terms is S_max(d).
            # Condition: remaining - 1/d <= S_max(d)
            
            d = start_d
            while d <= 10000000:
                # Check if 1/d <= remaining
                if Fraction(1, d) > remaining:
                    break # Since d increases, 1/d decreases, so this won't happen if we started correctly?
                    # Wait, start_d ensures 1/start_d <= remaining. So for larger d, 1/d is smaller.
                    # So this condition is always true for d >= start_d.
                
                # Check upper bound constraint
                # S_max = sum(1/i for i in range(d+1, d+terms_left))
                s_max = 0
                for i in range(terms_left - 1):
                    s_max += Fraction(1, d + 1 + i)
                
                if remaining - Fraction(1, d) > s_max:
                    # Even if we take the smallest possible remaining denominators, the sum is too large?
                    # No, S_max is the MAX sum.
                    # If required sum > Max possible sum, it's impossible.
                    # As d increases, 1/d decreases, so remaining - 1/d increases.
                    # Also S_max decreases as d increases.
                    # So if it's impossible now, it will be impossible for larger d.
                    # Wait.
                    # Let f(d) = (remaining - 1/d) - S_max(d).
                    # We need f(d) <= 0.
                    # If f(d) > 0, we need to increase d?
                    # As d increases, 1/d decreases, so (remaining - 1/d) increases.
                    # S_max(d) decreases.
                    # So f(d) increases.
                    # So if f(d) > 0, it will stay > 0 for larger d.
                    # So we can BREAK?
                    # No, wait.
                    # We need remaining - 1/d to be <= S_max(d).
                    # If it's > S_max(d), it's impossible.
                    # And for larger d, it becomes even more impossible.
                    # So yes, break.
                    break
                
                # Also check if remaining - 1/d is too small to be achieved?
                # Min sum of terms_left-1 terms > d is approx (terms_left-1)/(d+terms_left-1)?
                # Actually, the sum can be arbitrarily small by choosing large denominators.
                # So no lower bound pruning on d from this side, except d <= 10^7.
                
                new_remaining = remaining - Fraction(1, d)
                res = backtrack(new_remaining, d + 1, terms_left - 1, chosen + [d])
                if res is not None:
                    return res
                
                d += 1
                
            return None

        return backtrack(target, 5, n, [])

    for n in range(5, 257):
        result = solve(n)
        if result:
            return result
            
    raise RuntimeError("no representation found")
