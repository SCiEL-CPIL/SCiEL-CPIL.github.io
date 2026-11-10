import sys
import time
import numpy as np

def construct():
    n = 5040
    target = 4979
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Generate divisors
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    divisors.sort(reverse=True)
    N = len(divisors)
    
    INF = 10**9
    
    # Using Numpy for speed
    dp = np.full((N + 1, target + 1), INF, dtype=np.int32)
    choice = np.zeros((N, target + 1), dtype=np.uint8)
    
    dp[N, 0] = 0
    
    for i in range(N - 1, -1, -1):
        if time.time() > deadline:
            break
        
        d = divisors[i]
        dp_next = dp[i + 1]
        dp_curr = dp[i]
        
        # Initialize with skip values
        dp_curr[:] = dp_next[:]
        
        # Calculate take values
        if d <= target:
            take_part = 1 + dp_next[:target + 1 - d]
            skip_part = dp_next[d:]
            
            take_better = take_part < skip_part
            
            dp_curr[d:] = np.where(take_better, take_part, skip_part)
            choice[i, d:] = take_better.astype(np.uint8)
            
    if dp[0, target] == INF:
        return []
        
    result = []
    s = target
    for i in range(N):
        if choice[i, s] == 1:
            result.append(divisors[i])
            s -= divisors[i]
            
    return result
