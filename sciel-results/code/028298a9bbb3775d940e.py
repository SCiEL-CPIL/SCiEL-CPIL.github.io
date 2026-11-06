import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    n = 5040
    target = 4979
    
    # Generate divisors
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    
    # Filter and sort
    divs = [d for d in divs if d <= target]
    divs.sort(reverse=True)
    
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # parent[s] = (prev_sum, divisor)
    parent = [None] * (target + 1)
    
    # Local variables for speed
    dp_local = dp
    parent_local = parent
    target_local = target
    INF_local = INF
    
    for d in divs:
        if time.time() > deadline:
            break
            
        # Update from target down to d
        # Using local variables for faster access
        for s in range(target_local, d - 1, -1):
            prev_val = dp_local[s - d]
            if prev_val != INF_local:
                new_val = prev_val + 1
                if new_val < dp_local[s]:
                    dp_local[s] = new_val
                    parent_local[s] = (s - d, d)
        
        # Early exit if we found a solution?
        # No, we need to minimize the count.
        # A solution found with d might not be the minimum if a later d (smaller value, but we are going down?)
        # We are iterating divisors. The order doesn't affect the final min count in 0/1 DP.
        # However, if we found a solution, we can't stop immediately because a different combination
        # using fewer divisors might be found later?
        # Actually, in 0/1 DP, we process each item once.
        # The state dp[s] holds the min count using items processed so far.
        # So we must process all items to guarantee the global minimum.
        
    if dp_local[target_local] == INF_local:
        return []
        
    # Reconstruct
    current = target_local
    result = []
    while current > 0:
        prev, d = parent_local[current]
        result.append(d)
        current = prev
        
    return result
