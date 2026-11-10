import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Fast DFS to try to find a solution with fewer terms (7 or 8)
    # with strong pruning using exact rational arithmetic
    
    def dfs(rem_num, rem_den, min_d, current, k_target, time_limit):
        if time.time() > time_limit:
            return None
        
        k_left = k_target - len(current)
        
        if k_left == 0:
            if rem_num == 0:
                return current[:]
            return None
        
        if k_left == 1:
            # Need rem_num/rem_den = 1/d => d = rem_den/rem_num
            if rem_num != 0 and rem_den % rem_num == 0:
                d = rem_den // rem_num
                if d >= min_d and d >= 5 and d not in current:
                    return current + [d]
            return None
        
        # Lower bound: 1/d <= rem => d >= rem_den/rem_num
        d_min = max(min_d, 5, (rem_den + rem_num - 1) // rem_num)
        
        # Upper bound: need k_left terms each >= 1/d, sum = rem
        # So k_left/d >= rem => d <= k_left * rem_den / rem_num
        d_max = (k_left * rem_den) // rem_num
        
        if d_max < d_min:
            return None
        
        # Pruning: max possible sum with k_left terms starting from d_min
        # 1/d_min + 1/(d_min+1) + ... + 1/(d_min+k_left-1) < rem => impossible
        max_sum_num, max_sum_den = 0, 1
        for i in range(k_left):
            d_i = d_min + i
            # Add 1/d_i
            new_num = max_sum_num * d_i + max_sum_den
            new_den = max_sum_den * d_i
            g = gcd(new_num, new_den)
            max_sum_num = new_num // g
            max_sum_den = new_den // g
        
        # Check max_sum < rem
        if max_sum_num * rem_den < rem_num * max_sum_den:
            return None
        
        current_set = set(current)
        
        for d in range(d_min, d_max + 1):
            if d in current_set:
                continue
            
            # new_rem = rem - 1/d = (rem_num*d - rem_den) / (rem_den*d)
            new_num = rem_num * d - rem_den
            if new_num < 0:
                break
            if new_num == 0:
                continue  # Need more terms, can't use up the sum
            
            new_den = rem_den * d
            g = gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g
            
            result = dfs(new_num, new_den, d + 1, current + [d], k_target, time_limit)
            if result is not None:
                return result
        
        return None
    
    # Try to find a 7-term solution quickly (2 seconds)
    if time.time() < deadline - 45:
        result7 = dfs(1, 1, 5, [], 7, start_time + 2.0)
        if result7 is not None:
            # Verify
            total = sum(Fraction(1, d) for d in result7)
            if total == 1 and len(result7) == 7 and len(set(result7)) == 7 and all(5 <= d <= 10000000 for d in result7):
                return result7
    
    # Try to find an 8-term solution quickly (3 seconds)
    if time.time() < deadline - 40:
        result8 = dfs(1, 1, 5, [], 8, start_time + 3.0)
        if result8 is not None:
            total = sum(Fraction(1, d) for d in result8)
            if total == 1 and len(result8) == 8 and len(set(result8)) == 8 and all(5 <= d <= 10000000 for d in result8):
                return result8
    
    # Fallback: DP over divisors of 2520 (guaranteed to find a solution)
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d != 0:
            continue
        w = target // d
        for total, values in list(reached.items()):
            new_total = total + w
            if new_total <= target and new_total not in reached:
                reached[new_total] = values + [d]
        if target in reached:
            sol = reached[target]
            if len(sol) <= 256 and all(5 <= d <= 10000000 for d in sol):
                return sol
    
    raise RuntimeError("no solution found")
