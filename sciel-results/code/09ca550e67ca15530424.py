import time

def construct():
    start_time = time.time()
    target = 479001600 # 12!
    
    # M is the maximum factor allowed
    # We try M from 13 upwards
    # Lower bound for M:
    # If we use k factors, the max factor is at least the k-th smallest factor.
    # Just start from 13.
    
    def dfs(rem, last, count, path):
        nonlocal start_time
        if time.time() - start_time > 50: # Safety margin
            return None
        
        if rem == 1:
            return path[:]
        
        if count >= 12:
            return None
            
        # Pruning: 
        # The smallest possible next factor is last + 1.
        # If rem is less than last + 1, we can't pick any factor (unless rem==1 which is handled).
        if rem < last + 1:
            return None
            
        # Another prune: 
        # The maximum product we can form with the remaining slots (12 - count)
        # using the largest available numbers (M, M-1, ...) is an upper bound.
        # But we need to hit exactly `rem`.
        # A better prune: if rem is not divisible by any number in (last, M], return None.
        # We can check this quickly.
        
        # Iterate candidates
        for c in range(last + 1, M + 1):
            if rem % c == 0:
                path.append(c)
                res = dfs(rem // c, c, count + 1, path)
                if res is not None:
                    return res
                path.pop()
                
        return None

    # Estimate a reasonable starting M
    # 13*14*15*16*17*18*19*20 is ~ 1.3e9 > 4.8e8
    # 13*14*15*16*17*18*19 is ~ 9.6e7 < 4.8e8
    # So we probably need around 8-9 factors.
    # The max factor will likely be around 20-30.
    
    M = 13
    while M <= 100: # Upper bound safety
        if time.time() - start_time > 55:
            break
        path = []
        res = dfs(target, 12, 0, path)
        if res is not None:
            return res
        M += 1
        
    # Fallback if not found (should not happen)
    return [target]
