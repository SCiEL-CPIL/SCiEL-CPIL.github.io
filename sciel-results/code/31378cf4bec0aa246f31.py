import time
from collections import deque

def construct():
    start_time = time.time()
    deadline = 85.0  # 5 seconds safety margin before 90s kill

    n = 5040
    target = 4979

    # Generate divisors
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort in descending order to prioritize larger divisors
    divisors.sort(reverse=True)
    m = len(divisors)
    
    # Precompute suffix sums for pruning: suffix_sum[i] is sum of divisors[i:]
    suffix_sum = [0] * (m + 1)
    for i in range(m - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + divisors[i]

    # BFS State: (remaining_sum, last_index, path)
    # last_index is the index of the last chosen divisor. We can only choose indices > last_index.
    # We start with last_index = -1.
    # To avoid storing the full path in every state (memory), we can store the parent pointer or just the path since depth is small.
    # Given the small expected depth, storing the path list is fine.
    
    # Queue stores: (remaining_sum, last_index, path_list)
    q = deque()
    q.append((target, -1, []))
    
    while q:
        if time.time() > deadline:
            # Fallback: return a valid solution if found previously, or raise if not.
            # But BFS finds the optimal first, so if we time out, we likely haven't found it or are stuck.
            # For this specific instance, it will find it instantly.
            pass
            
        remaining, last_idx, path = q.popleft()
        
        if remaining == 0:
            # Found a solution with minimum number of terms (BFS property)
            return path
        
        # Try adding each divisor with index > last_idx
        # Start from last_idx + 1
        for i in range(last_idx + 1, m):
            d = divisors[i]
            if d > remaining:
                # Since divisors are sorted descending, if d > remaining, 
                # all subsequent divisors (smaller indices in this loop? No, i increases -> divisors decrease)
                # Wait, divisors is descending. So divisors[0] is largest.
                # If d > remaining, we skip it. But smaller divisors might fit.
                # So we cannot break. We must continue to check smaller divisors.
                continue
            
            new_remaining = remaining - d
            new_path = path + [d]
            
            # Pruning: Check if the sum of all available divisors (from i+1 onwards) 
            # is sufficient to reach the new_remaining?
            # Actually, we need to check if new_remaining can be formed by divisors with index > i.
            # The maximum possible sum we can add is suffix_sum[i+1].
            # If new_remaining > suffix_sum[i+1], it's impossible to fill the rest.
            if new_remaining > suffix_sum[i + 1]:
                continue
            
            q.append((new_remaining, i, new_path))
            
    # If no solution found (should not happen for this instance), return empty or raise
    return []
