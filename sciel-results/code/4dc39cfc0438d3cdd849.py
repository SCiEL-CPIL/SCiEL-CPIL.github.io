import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Try to find a representation with k terms
    for k in range(5, 257):
        if time.time() > deadline:
            break
            
        def dfs(depth, prev_den, num, den):
            if time.time() > deadline:
                return None
            
            if depth == k:
                if num == 0:
                    return []
                return None
            
            m = k - depth
            
            # min_d: 1/d <= num/den => d >= ceil(den/num)
            # Also d > prev_den
            min_d = max(prev_den + 1, (den + num - 1) // num)
            
            # max_d: num/den <= m/d => d <= m * den / num
            max_d = (m * den) // num
            
            if min_d > max_d:
                return None
            
            # If m == 1, we need 1/d = num/den => d = den/num
            if m == 1:
                if num == 1 and den >= min_d and den <= max_d:
                    return [den]
                return None
            
            for d in range(min_d, max_d + 1):
                # new_rem = num/den - 1/d = (num*d - den) / (den*d)
                new_num = num * d - den
                new_den = den * d
                
                if new_num < 0:
                    break
                
                # Pruning: new_rem <= (m-1)/(d+1)
                # new_num / new_den <= (m-1) / (d+1)
                # new_num * (d+1) <= new_den * (m-1)
                if new_num * (d + 1) > new_den * (m - 1):
                    break
                
                # Reduce fraction
                g = math.gcd(new_num, new_den)
                if g > 1:
                    new_num //= g
                    new_den //= g
                    
                res = dfs(depth + 1, d, new_num, new_den)
                if res is not None:
                    return [d] + res
            
            return None

        result = dfs(0, 4, 1, 1)
        if result is not None:
            return result
            
    # Fallback to the original DP if DFS fails or times out
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
