import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Known valid 8-term solution with distinct denominators >= 5
    # Verify: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/12 + 1/14
    # Actually let me use a verified one.
    # A known solution: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 doesn't work.
    # Let's use the DFS to find one, but also have a hardcoded fallback.
    
    # Hardcoded valid solution (will be verified)
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/12 + 1/13 + ... no, need exactly 8.
    # Let me compute: we need 8 distinct denominators >= 5 summing to 1.
    # One known minimal solution: [5, 6, 7, 8, 9, 10, 12, 14] - let me verify
    # Sum = 1/5+1/6+1/7+1/8+1/9+1/10+1/12+1/14
    # LCM(5,6,7,8,9,10,12,14) = 2^3*3^2*5*7 = 2520
    # 2520/5=504, 2520/6=420, 2520/7=360, 2520/8=315, 2520/9=280, 2520/10=252, 2520/12=210, 2520/14=180
    # Sum = 504+420+360+315+280+252+210+180 = 2521, not 2520.
    # So that's not valid.
    
    # Let me try [5, 6, 7, 8, 9, 10, 11, 12]:
    # LCM = 27720
    # 27720/5=5544, /6=4620, /7=3960, /8=3465, /9=3080, /10=2772, /11=2520, /12=2310
    # Sum = 5544+4620+3960+3465+3080+2772+2520+2310 = 28271, not 27720.
    
    # I'll rely on the DFS to find a valid solution, starting from k=8.
    
    def dfs(remaining_num, remaining_den, min_denom, current, max_depth):
        if time.time() > deadline:
            return None
        
        if remaining_num == 0:
            if len(current) == max_depth:
                return current
            return None
        
        if len(current) >= max_depth:
            return None
        
        terms_left = max_depth - len(current)
        
        # If remaining is a unit fraction and we have exactly one term left
        if terms_left == 1:
            if remaining_den == 1:
                d = remaining_num
                if d >= min_denom and d >= 5 and d not in current:
                    return current + [d]
            return None
        
        # If remaining is a unit fraction and we can add it with more terms to spare
        # This shouldn't happen in a minimal search, but handle it
        if remaining_den == 1:
            d = remaining_num
            if d >= min_denom and d >= 5 and d not in current:
                return current + [d]
        
        # 1/d <= remaining, so d >= ceil(1/remaining) = ceil(den/num)
        # min_d: smallest d such that 1/d <= remaining
        # 1/d <= num/den => d >= den/num
        min_d = max(min_denom, 5, (remaining_den + remaining_num - 1) // remaining_num)
        
        # Upper bound: we need terms_left terms each <= 1/d_next, and their sum = remaining
        # So 1/d >= remaining / terms_left => d <= terms_left / remaining = terms_left * den / num
        max_d = (terms_left * remaining_den) // remaining_num
        
        if max_d < min_d:
            return None
        
        # Also, the sum of the largest possible remaining terms must be >= remaining
        # The maximum sum with terms_left distinct denominators >= min_d is:
        # 1/min_d + 1/(min_d+1) + ... + 1/(min_d+terms_left-1)
        # If this is < remaining, no solution
        max_possible_sum_num = 0
        max_possible_sum_den = 1
        for i in range(terms_left):
            d_i = min_d + i
            # Add 1/d_i to the sum
            new_num = max_possible_sum_num * d_i + max_possible_sum_den
            new_den = max_possible_sum_den * d_i
            g = gcd(new_num, new_den)
            max_possible_sum_num = new_num // g
            max_possible_sum_den = new_den // g
        
        # Check if max_possible_sum < remaining
        # max_possible_sum_num/max_possible_sum_den < remaining_num/remaining_den
        # <=> max_possible_sum_num * remaining_den < remaining_num * max_possible_sum_den
        if max_possible_sum_num * remaining_den < remaining_num * max_possible_sum_den:
            return None
        
        # Also check minimum possible sum: 1/max_d + 1/(max_d-1) + ... but this is complex
        # Skip for now
        
        current_set = set(current)
        
        for d in range(min_d, min(max_d + 1, 10000001)):
            if d in current_set:
                continue
            
            # new_remaining = remaining - 1/d = (num*d - den) / (den*d)
            new_num = remaining_num * d - remaining_den
            if new_num < 0:
                break
            if new_num == 0:
                # We've used up the remaining sum, but we need more terms
                # This is invalid since we need exactly max_depth terms
                continue
            new_den = remaining_den * d
            g = gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g
            
            result = dfs(new_num, new_den, d + 1, current + [d], max_depth)
            if result is not None:
                return result
        
        return None

    # Start from k=8 since we know k<8 is impossible
    for max_terms in range(8, 9):
        result = dfs(1, 1, 5, [], max_terms)
        if result is not None:
            total = sum(Fraction(1, d) for d in result)
            if total == 1 and len(result) == len(set(result)) and all(5 <= d <= 10000000 for d in result):
                return result

    # Fallback: return a hardcoded valid solution
    # After extensive search, a known valid 8-term solution:
    # Let me use the DP approach as fallback
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            sol = reached[target]
            if len(sol) <= 256 and all(5 <= d <= 10000000 for d in sol):
                return sol
    
    raise RuntimeError("no baseline representation")
