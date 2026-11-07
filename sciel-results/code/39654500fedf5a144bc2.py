import time
import random

def _try_order(order, max_val=256, max_size=60, deadline=None):
    """Try adding elements in the given order, return the set found."""
    dp = [set() for _ in range(max_size + 1)]
    dp[0].add(0)
    chosen = []
    
    for x in order:
        if len(chosen) >= max_size:
            break
        if deadline is not None and time.time() > deadline:
            break
        
        valid = True
        new_sums = [set() for _ in range(max_size + 1)]
        
        for k in range(len(chosen), -1, -1):
            if dp[k]:
                new_sums[k+1] = {s + x for s in dp[k]}
                if dp[k+1] and new_sums[k+1] & dp[k+1]:
                    valid = False
                    break
        
        if valid:
            chosen.append(x)
            for k in range(len(chosen), 0, -1):
                dp[k] = dp[k] | new_sums[k]
    
    return chosen

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    max_val = 256
    max_size = 60
    
    best = []
    
    # Try increasing order
    order1 = list(range(1, max_val + 1))
    result1 = _try_order(order1, max_val, max_size, deadline)
    if len(result1) > len(best):
        best = result1
    
    if time.time() > deadline:
        return [int(x) for x in best]
    
    # Try decreasing order
    order2 = list(range(max_val, 0, -1))
    result2 = _try_order(order2, max_val, max_size, deadline)
    if len(result2) > len(best):
        best = result2
    
    if time.time() > deadline:
        return [int(x) for x in best]
    
    # Try some random orders
    random.seed(42)
    for trial in range(100):
        if time.time() > deadline:
            break
        
        order = list(range(1, max_val + 1))
        random.shuffle(order)
        result = _try_order(order, max_val, max_size, deadline)
        if len(result) > len(best):
            best = result
    
    return [int(x) for x in best]
