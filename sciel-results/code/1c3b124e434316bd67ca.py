import time
import sys

def construct():
    start_time = time.time()
    deadline = 60.0  # 60s limit, keep safety margin
    
    # Try to find a solution with k terms
    # k=1, 2, 3, 4 are likely impossible with d>=5
    # We will try k=5, 6, ...
    
    # To ensure we return something, we can have a fallback.
    # But the task implies a solution exists.
    
    # DFS for Egyptian Fraction
    # State: remaining_sum, min_denom, current_list
    # Target: remaining_sum = 0
    
    def dfs(remaining_num, remaining_den, min_denom, current_list, max_depth, deadline):
        if time.time() > deadline:
            return None
        if remaining_num == 0:
            return current_list
        if len(current_list) == max_depth:
            if remaining_num == 0:
                return current_list
            return None
        
        # Pruning: if remaining_num / remaining_den is too large to be formed by max_depth - len(current_list) terms
        # Max possible sum with m terms starting from min_denom is roughly m * (1/min_denom)
        # Actually, 1/min_denom + 1/(min_denom+1) + ...
        # Simpler bound: 1/x + 1/y + ... < m/x if all are >= x
        # So if remaining > m * (1/min_denom), it's impossible? No, 1/x + 1/(x+1) < 2/x.
        # Sum of 1/i for i=min_denom to min_denom+m-1 is an upper bound.
        # Let's use a simpler bound: if remaining_num * min_denom > remaining_den * max_terms_left:
        # 1/min_denom + ... < max_terms_left / min_denom
        # So if remaining > max_terms_left / min_denom, return None.
        
        max_terms_left = max_depth - len(current_list)
        if remaining_num * min_denom > remaining_den * max_terms_left:
            return None
            
        # Lower bound for next denominator
        # 1/d <= remaining => d >= 1/remaining
        # Also d >= min_denom
        # Also 1/d + ... >= remaining
        # 1/d + 1/(d+1) + ... >= remaining
        # Approx: 1/d + (max_terms_left - 1) / (d+1) >= remaining?
        # Simpler: 1/d <= remaining => d >= 1/remaining
        # Also, the sum of the remaining max_terms_left terms must be at least remaining.
        # The largest possible sum for max_terms_left terms with min denominator d is 1/d + 1/(d+1) + ...
        # We can iterate d from lower_bound to upper_bound.
        
        # Upper bound for d:
        # 1/d must be at least remaining / max_terms_left
        # So d <= max_terms_left / remaining
        # Also d must be >= min_denom
        
        lower_d = max(min_denom, (remaining_den + remaining_num - 1) // remaining_num) # ceil(1/remaining)
        upper_d = (remaining_den * max_terms_left) // remaining_num
        
        # Iterate d
        for d in range(lower_d, upper_d + 1):
            if remaining_num * d <= remaining_den:
                new_num = remaining_num * d - remaining_den
                new_den = remaining_den * d
                # Reduce fraction?
                # gcd(new_num, new_den)
                g = 1
                a, b = new_num, new_den
                while b:
                    a, b = b, a % b
                g = a
                if g > 1:
                    new_num //= g
                    new_den //= g
                    
                res = dfs(new_num, new_den, d + 1, current_list + [d], max_depth, deadline)
                if res is not None:
                    return res
        return None

    # Try k=5
    for k in range(5, 10):
        res = dfs(1, 1, 5, [], k, deadline - 5) # 5s margin
        if res is not None:
            return res
            
    # Fallback to the original DP if DFS fails or times out
    # But DFS should find a solution for k=5 or 6 quickly.
    # 1/5 + 1/6 + 1/7 + 1/14 + 1/210 = 1?
    # 1/5 + 1/6 = 11/30
    # 11/30 + 1/7 = 77/210 + 30/210 = 107/210
    # 107/210 + 1/14 = 107/210 + 15/210 = 122/210 = 61/105
    # 61/105 + 1/210 = 122/210 + 1/210 = 123/210 != 1.
    
    # Known solution: 1 = 1/5 + 1/6 + 1/7 + 1/14 + 1/210 is wrong.
    # 1 = 1/5 + 1/6 + 1/7 + 1/14 + 1/210?
    # Sum = (42 + 35 + 30 + 15 + 1) / 210 = 123/210.
    
    # Let's just let the DFS find it.
    
    # If DFS fails, use the original method
    target, reached = 2520, {0: []}
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
