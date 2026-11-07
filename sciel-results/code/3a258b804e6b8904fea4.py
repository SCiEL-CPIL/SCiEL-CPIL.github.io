import time

def construct():
    max_val = 256
    target_size = 60
    start_time = time.time()
    time_limit = 55.0
    
    best_set = []
    
    def is_valid_add(dp, x, current_size):
        # Check if adding x is valid
        # Condition: (S_k + x) & S_m == 0 for all k, m with m != k+1
        # S_k is dp[k]
        
        for k in range(current_size + 1):
            if dp[k] == 0:
                continue
            shifted = dp[k] << x
            for m in range(current_size + 1):
                if m == k + 1:
                    continue
                if (shifted & dp[m]) != 0:
                    return False
        return True

    def backtrack(path, dp, current_size, target_size, used):
        nonlocal best_set
        if len(path) > len(best_set):
            best_set = path[:]
        
        if len(path) == target_size:
            return True
        
        if time.time() - start_time > time_limit:
            return False
            
        # Heuristic: Try larger numbers first
        for x in range(max_val, 0, -1):
            if x in used:
                continue
            
            if is_valid_add(dp, x, current_size):
                new_dp = [0] * (target_size + 1)
                for k in range(target_size + 1):
                    new_dp[k] = dp[k]
                    if k > 0:
                        new_dp[k] |= (dp[k-1] << x)
                
                used.add(x)
                if backtrack(path + [x], new_dp, current_size + 1, target_size, used):
                    used.remove(x)
                    return True
                used.remove(x)
        
        return False

    initial_dp = [0] * (target_size + 1)
    initial_dp[0] = 1
    used = set()
    
    # Run backtracking
    backtrack([], initial_dp, 0, target_size, used)
    
    if not best_set:
        # Fallback
        return [1]
        
    return best_set
