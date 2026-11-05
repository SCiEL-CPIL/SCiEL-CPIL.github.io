import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = 55.0  # safety margin before 60s limit
    
    def elapsed():
        return time.time() - start_time
    
    def time_ok():
        return elapsed() < deadline
    
    # k=1..8 impossible, k=9 is minimum
    # Use DFS with integer arithmetic and strong pruning
    
    def dfs(rem_num, rem_den, min_d, depth, max_depth, path, memo):
        if not time_ok():
            return None
        if rem_num == 0:
            return path
        if depth == max_depth:
            return None
        
        terms_left = max_depth - depth
        
        # Pruning 1: Upper bound - rem must be <= sum of 1/(min_d) + 1/(min_d+1) + ...
        # Approximate: rem <= terms_left / min_d
        # rem_num/rem_den <= terms_left / min_d
        # rem_num * min_d <= rem_den * terms_left
        if rem_num * min_d > rem_den * terms_left:
            return None
        
        # Pruning 2: Lower bound - rem must be >= terms_left / 10^7
        # rem_num/rem_den >= terms_left / 10^7
        # rem_num * 10^7 >= rem_den * terms_left
        if rem_num * 10**7 < rem_den * terms_left:
            return None
        
        # Memoization key
        key = (rem_num, rem_den, min_d, terms_left)
        if key in memo:
            return None
        
        memo.add(key)
        
        # d_min: 1/d <= rem => d >= 1/rem => d >= rem_den/rem_num
        d_min = max(min_d, (rem_den + rem_num - 1) // rem_num)
        
        # d_max: 1/d + (terms_left-1)/(d+1) >= rem (loose bound)
        # Simpler: terms_left/d >= rem => d <= terms_left/rem
        d_max = (terms_left * rem_den) // rem_num
        
        if d_min > d_max:
            return None
        
        # Try denominators from d_min to d_max
        for d in range(d_min, d_max + 1):
            if not time_ok():
                return None
            
            # new_rem = rem - 1/d = (rem_num*d - rem_den) / (rem_den*d)
            new_num = rem_num * d - rem_den
            if new_num < 0:
                break  # 1/d > rem, no more valid
            
            new_den = rem_den * d
            
            # Reduce fraction
            g = gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g
            
            res = dfs(new_num, new_den, d + 1, depth + 1, max_depth, path + [d], memo)
            if res is not None:
                return res
        
        return None
    
    # Try k=9 first
    for k in range(9, 13):
        if not time_ok():
            break
        memo = set()
        res = dfs(1, 1, 5, 0, k, [], memo)
        if res is not None:
            return res
    
    # Fallback: subset sum DP with larger common denominator
    # Try with LCM of small numbers
    from math import lcm
    # Use a common denominator that's a multiple of many small numbers
    # 2520 = lcm(1..10), but we need denominators >= 5
    # Let's try with a larger target
    target = 2520 * 13  # 32760, includes factor 13
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d != 0:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    
    # Last resort: try with even larger target
    target2 = 2520 * 13 * 17  # 556920
    reached2 = {0: []}
    for d in range(5, target2 + 1):
        if target2 % d != 0:
            continue
        w = target2 // d
        for total, values in list(reached2.items()):
            if total + w <= target2 and total + w not in reached2:
                reached2[total + w] = values + [d]
        if target2 in reached2:
            return reached2[target2]
    
    raise RuntimeError("No solution found")
