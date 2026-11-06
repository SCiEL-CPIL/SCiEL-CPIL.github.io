"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from bisect import bisect_right

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s hard kill
    
    # ---- Sieve of primes up to 10000 ----
    N = 10000
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(N + 1) if is_prime[i]]
    K = len(primes)
    
    # ---- Dynamic Programming over pairs of primes ----
    # dp[i][j] = length of longest valid prime subsequence ending with primes[i], primes[j] (i < j)
    # where all gaps in the subsequence are non-decreasing.
    # 
    # Recurrence: dp[i][j] = 1 + max(dp[h][i] for h < i where primes[i]-primes[h] <= primes[j]-primes[i])
    # Base: dp[i][j] = 2 if no valid h exists (just two primes)
    #
    # Note: dp[h][i] already includes both primes[h] and primes[i], so adding primes[j]
    # increases the length by 1, not 2.
    
    dp = [[0] * K for _ in range(K)]
    best = 2
    best_i = -1
    best_j = -1
    
    for i in range(K):
        if time.time() > deadline:
            break
        if i == 0:
            # No h < 0, so dp[0][j] = 2 for all j > 0
            for j in range(1, K):
                dp[0][j] = 2
            continue
        
        # Build list of (gap, dp_val) for h < i
        list_i = []
        for h in range(i):
            gap = primes[i] - primes[h]
            dp_val = dp[h][i]
            if dp_val > 0:
                list_i.append((gap, dp_val))
        
        if not list_i:
            for j in range(i + 1, K):
                dp[i][j] = 2
            continue
        
        # Sort by gap
        list_i.sort(key=lambda x: x[0])
        
        # Extract gaps and compute prefix max
        gaps = [x[0] for x in list_i]
        prefix_max = [0] * len(list_i)
        prefix_max[0] = list_i[0][1]
        for idx in range(1, len(list_i)):
            prefix_max[idx] = max(prefix_max[idx - 1], list_i[idx][1])
        
        for j in range(i + 1, K):
            if time.time() > deadline:
                break
            target = primes[j] - primes[i]
            # Find rightmost index where gap <= target
            idx = bisect_right(gaps, target) - 1
            if idx >= 0:
                # dp[i][j] = 1 + max(dp[h][i]) for valid h
                dp[i][j] = 1 + prefix_max[idx]
            else:
                dp[i][j] = 2
            
            if dp[i][j] > best:
                best = dp[i][j]
                best_i = i
                best_j = j
    
    if best <= 2:
        return primes[:2]
    
    # Backtrack to reconstruct the sequence
    sequence = []
    i, j = best_i, best_j
    sequence.append(primes[j])
    sequence.append(primes[i])
    
    while len(sequence) < best:
        target_gap = primes[j] - primes[i]
        found = False
        for h in range(i):
            if dp[h][i] == dp[i][j] - 1:
                if primes[i] - primes[h] <= target_gap:
                    sequence.append(primes[h])
                    i, j = h, i
                    found = True
                    break
        if not found:
            break
    
    sequence.reverse()
    
    # Verify the sequence
    for k in range(len(sequence) - 1):
        if sequence[k + 1] <= sequence[k]:
            return primes[:2]
        if k > 0:
            gap_curr = sequence[k + 1] - sequence[k]
            gap_prev = sequence[k] - sequence[k - 1]
            if gap_curr < gap_prev:
                return primes[:2]
    
    return sequence

# EVOLVE-BLOCK-END
