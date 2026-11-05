from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = 50.0 # Leave 10s margin for evaluation overhead

    def dfs(rem, last, depth, k, path):
        if time.time() > deadline:
            return None
        if depth == k:
            if rem == 0:
                return path[:]
            return None

        if rem <= 0:
            return None

        num = rem.numerator
        den = rem.denominator

        # Lower bound for d
        # 1/d <= rem => d >= 1/rem = den/num
        min_d = (den + num - 1) // num
        if min_d <= last:
            min_d = last + 1

        # Upper bound for d
        # We need to fill rem - 1/d with (k - depth - 1) terms.
        # The max sum of (k - depth - 1) terms with denominators > d is (k - depth - 1) / (d + 1).
        # So rem - 1/d <= (k - depth - 1) / (d + 1)
        # rem <= 1/d + (k - depth - 1) / (d + 1)
        # rem * d * (d + 1) <= (d + 1) + (k - depth - 1) * d
        # rem * (d^2 + d) <= d + 1 + (k - depth - 1) * d
        # rem * d^2 + rem * d <= (k - depth) * d + 1
        # rem * d^2 + (rem - (k - depth)) * d - 1 <= 0
        
        # Simpler bound: rem <= (k - depth) / d  => d <= (k - depth) / rem
        # This is looser but easier.
        # Let's use the tighter one if possible.
        
        # Also, the minimum sum of (k - depth - 1) terms with denominators >= d+1
        # is not a strong lower bound for pruning d, but the upper bound on remaining sum is.
        
        max_d = (k - depth) * den // num
        if max_d < min_d:
            return None

        # Iterate d
        for d in range(min_d, max_d + 1):
            if time.time() > deadline:
                return None
            
            # Check if 1/d <= rem
            if d < den / num: # Using float for quick check? No, use integer
                # d * num < den
                if d * num < den:
                    continue # 1/d > rem, so d is too small? 
                    # Wait, min_d ensures 1/d <= rem. 
                    # So d >= ceil(den/num). 
                    # If d < den/num, then 1/d > rem. 
                    # But min_d is ceil(den/num), so d >= min_d implies d >= den/num (approx).
                    # Actually if d = ceil(den/num), then d >= den/num.
                    # So 1/d <= num/den = rem.
                    # So this check is redundant if min_d is correct.
                    pass

            new_rem = rem - Fraction(1, d)
            if new_rem < 0:
                break # d is too small? No, d increases, 1/d decreases.
                # If 1/d > rem, new_rem < 0.
                # Since d starts from min_d, 1/d <= rem.
                # As d increases, 1/d decreases, so new_rem increases?
                # No, rem is fixed. 1/d decreases.
                # So new_rem = rem - 1/d increases as d increases.
                # So if new_rem < 0, it means 1/d > rem.
                # But min_d ensures 1/d <= rem.
                # So new_rem should be >= 0.
                # Wait, if d < den/num, 1/d > rem.
                # But d >= min_d = ceil(den/num).
                # So d >= den/num.
                # So 1/d <= num/den = rem.
                # So new_rem >= 0.
                # So this break is never triggered?
                # Unless min_d calculation is wrong.
                # min_d = ceil(den/num).
                # If den/num is integer, min_d = den/num. 1/d = num/den = rem. new_rem = 0.
                # If den/num is not integer, min_d > den/num. 1/d < rem. new_rem > 0.
                # So new_rem is always >= 0.
                
            # Additional pruning:
            # Can new_rem be represented by (k - depth - 1) terms?
            # Max sum of (k - depth - 1) terms with denominators > d is (k - depth - 1) / (d + 1).
            if new_rem > Fraction(k - depth - 1, d + 1):
                # We need to pick a larger d to make 1/d smaller, so new_rem larger?
                # No, if d increases, 1/d decreases, new_rem increases.
                # We need new_rem <= max_possible_remaining_sum.
                # Max possible remaining sum is (k - depth - 1) / (d + 1).
                # If new_rem > this, it's impossible.
                # But as d increases, new_rem increases and max_possible_remaining_sum decreases.
                # So if it's impossible now, it will be impossible for larger d too?
                # Let's check.
                # f(d) = new_rem - (k - depth - 1)/(d + 1)
                # new_rem = rem - 1/d.
                # f(d) = rem - 1/d - (k - depth - 1)/(d + 1)
                # As d increases, -1/d increases (less negative), and -(k...)/(d+1) increases (less negative).
                # So f(d) is increasing?
                # If f(d) > 0, we need a smaller new_rem?
                # No, we need new_rem <= max_sum.
                # If new_rem > max_sum, we fail.
                # If we increase d, new_rem increases (worse) and max_sum decreases (worse).
                # So if we fail at d, we will fail for all larger d.
                # So we can break?
                # Wait, min_d is the smallest d.
                # At min_d, 1/d is largest, so new_rem is smallest.
                # And max_sum is largest (since d is smallest, d+1 is smallest).
                # So if it fails at min_d, it might fail for all d?
                # Let's check the derivative.
                # Actually, if new_rem > max_sum at some d, it means the remaining sum is too large to be covered by the remaining terms.
                # To reduce new_rem, we need to INCREASE 1/d, i.e., DECREASE d.
                # But d cannot be smaller than min_d.
                # So if it fails at min_d, the search fails?
                # No, min_d is the lower bound for d.
                # If it fails at min_d, we cannot choose a smaller d.
                # So we should return None?
                # But we are iterating d from min_d upwards.
                # If it fails at min_d, does it fail for larger d?
                # Larger d -> smaller 1/d -> larger new_rem.
                # Larger d -> larger d+1 -> smaller max_sum.
                # So the gap (new_rem - max_sum) increases.
                # So if it's positive at min_d, it will be positive for all larger d.
                # So we can return None immediately if this condition holds at min_d?
                # Wait, the condition is `new_rem > Fraction(k - depth - 1, d + 1)`.
                # If true, we cannot fill the remaining sum.
                # So we should break? No, return None?
                # Yes, if it's impossible at the best case (smallest new_rem, largest max_sum), it's impossible for all other cases.
                return None

            path.append(d)
            res = dfs(new_rem, d, depth + 1, k, path)
            if res:
                return res
            path.pop()
        return None

    # Try k=10 first? No, try 8, 9, 10.
    # But 8 and 9 are likely impossible and will take time to prove.
    # We can add a quick check or just let it run with tight time.
    
    for k in range(8, 14):
        if time.time() > deadline:
            break
        res = dfs(Fraction(1, 1), 4, 0, k, [])
        if res:
            return res

    # Fallback: Greedy
    target = Fraction(1, 1)
    denominators = []
    last = 4
    while target > 0 and time.time() < deadline:
        d = (target.denominator + target.numerator - 1) // target.numerator
        if d <= last:
            d = last + 1
        while Fraction(1, d) > target:
            d += 1
        denominators.append(d)
        target -= Fraction(1, d)
        last = d
        if len(denominators) > 256:
            break
    return denominators
