"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit

    def dfs(rem_num, rem_den, last_d, current_set, max_depth):
        if time.time() > deadline:
            return None
        
        if rem_num == 0:
            return current_set if current_set else None
        
        if rem_num < 0 or rem_den <= 0:
            return None
        
        if len(current_set) >= max_depth:
            return None
        
        # 1/d <= rem_num/rem_den  => d >= rem_den/rem_num
        min_d = rem_den // rem_num
        if rem_den % rem_num != 0:
            min_d += 1
        min_d = max(min_d, last_d + 1, 5)
        
        # Limit the search range to avoid infinite loops or excessive branching
        # For small k, denominators are usually small.
        # We can expand this if needed, but 1000 is a safe upper bound for small k.
        max_d = min_d + 1000 
        
        for d in range(min_d, max_d):
            # 1/d must be <= rem
            if d * rem_num < rem_den:
                continue
            
            new_num = rem_num * d - rem_den
            new_den = rem_den * d
            
            if new_num == 0:
                if len(current_set) + 1 <= max_depth:
                    return current_set + [d]
                continue
            
            if new_num < 0:
                break # d is too small? No, d is increasing, so 1/d is decreasing.
                      # If 1/d > rem, then new_num < 0.
                      # As d increases, 1/d decreases, so new_num increases.
                      # So if new_num < 0, we should continue to larger d.
                      # Wait.
                      # rem = R. 1/d.
                      # If 1/d > R, then R - 1/d < 0.
                      # As d increases, 1/d decreases, so R - 1/d increases.
                      # So if new_num < 0, we should continue.
                      # The loop condition `if d * rem_num < rem_den: continue` handles this.
                      # So we don't break here.
            
            g = gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g
                
            res = dfs(new_num, new_den, d, current_set + [d], max_depth)
            if res:
                return res
        
        return None

    # Search for minimal k
    for k in range(5, 15):
        if time.time() > deadline:
            break
        res = dfs(1, 1, 4, [], k)
        if res:
            return res

    # Fallback to original DP if search fails
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
    
    raise RuntimeError("no solution found")
# EVOLVE-BLOCK-END
