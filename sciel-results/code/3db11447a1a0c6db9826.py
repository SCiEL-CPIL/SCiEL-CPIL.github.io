import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # 58s safety margin
    
    # Generate primes <= 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    n = len(primes)
    
    if n == 0:
        return []
    if n == 1:
        return primes
    
    dp = [1] * n
    prev = [-1] * n
    last_gap = [0] * n
    
    # DP to find longest subsequence
    for i in range(1, n):
        max_len = 1
        best_prev = -1
        best_gap = 0
        
        # Check all j < i
        for j in range(i):
            curr_gap = primes[i] - primes[j]
            prev_gap = last_gap[j]
            
            if curr_gap >= prev_gap:
                if dp[j] + 1 > max_len:
                    max_len = dp[j] + 1
                    best_prev = j
                    best_gap = curr_gap
                elif dp[j] + 1 == max_len:
                    # Tie-break: prefer smaller current gap (which means larger primes[j])
                    # to allow more room for future gaps
                    if curr_gap < best_gap:
                        best_prev = j
                        best_gap = curr_gap
        
        dp[i] = max_len
        prev[i] = best_prev
        last_gap[i] = best_gap
        
        if time.time() > deadline:
            break
            
    # Find the index with the maximum length
    max_len = max(dp)
    # If there are ties, we might want the one with the smallest last_gap?
    # Actually, the length is the score. Any max length is fine.
    # But to be consistent with the "smaller gap is better" heuristic, 
    # let's pick the one with the smallest last_gap among those with max_len.
    best_end = -1
    min_final_gap = float('inf')
    for i in range(n):
        if dp[i] == max_len:
            if last_gap[i] < min_final_gap:
                min_final_gap = last_gap[i]
                best_end = i
    if best_end == -1:
        best_end = 0
        
    # Reconstruct the sequence
    seq_indices = []
    curr = best_end
    while curr != -1:
        seq_indices.append(curr)
        curr = prev[curr]
        
    seq_indices.reverse()
    result = [primes[i] for i in seq_indices]
    
    return result
