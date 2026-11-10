import bisect
import time

def construct():
    # Deadline guard
    start_time = time.time()
    DEADLINE = 50  # Safety margin for 60s limit
    
    # 1. Sieve of Eratosthenes to get primes <= 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    N = len(primes)
    
    if N < 2:
        return primes
    
    # 2. DP Setup
    # dp[i][j] = max length of valid sequence ending with primes[i], primes[j]
    # prev[i][j] = index h such that sequence ends ... primes[h], primes[i], primes[j]
    # We use lists of lists for simplicity, N is small (~1229)
    dp = [[0] * N for _ in range(N)]
    prev = [[-1] * N for _ in range(N)]
    
    best_len = 0
    best_i = -1
    best_j = -1
    
    # Iterate i from 0 to N-2
    # For each i, we need dp[h][i] for h < i.
    # These are already computed because when outer loop was h, j could be i.
    
    for i in range(N - 1):
        # Check time
        if time.time() - start_time > DEADLINE:
            break
            
        # Precompute suffix maxes for dp[0..i-1][i]
        # We need max dp[h][i] for h in [k, i-1]
        # Let's build an array suf where suf[k] = (max_val, idx) for h in [k, i-1]
        # If i == 0, no h < 0.
        if i > 0:
            suf = [None] * i
            # Initialize last element
            last_h = i - 1
            suf[i-1] = (dp[last_h][i], last_h)
            
            for k in range(i - 2, -1, -1):
                val_next, idx_next = suf[k+1]
                val_curr = dp[k][i]
                if val_curr > val_next:
                    suf[k] = (val_curr, k)
                else:
                    suf[k] = (val_next, idx_next)
        else:
            suf = []
            
        # Iterate j from i+1 to N-1
        for j in range(i + 1, N):
            # Check time periodically
            if (j & 0xFF) == 0 and time.time() - start_time > DEADLINE:
                break
                
            gap = primes[j] - primes[i]
            # Condition: primes[i] - primes[h] <= gap
            # primes[h] >= primes[i] - gap = 2*primes[i] - primes[j]
            L = 2 * primes[i] - primes[j]
            
            # Find lower bound of L in primes[0..i-1]
            # bisect_left(a, x, lo, hi)
            # We search in primes[0:i]
            k = bisect.bisect_left(primes, L, 0, i)
            
            if k < i:
                # There is a valid h in [k, i-1]
                max_val, max_idx = suf[k]
                new_len = max_val + 1
                if new_len > 2:
                    dp[i][j] = new_len
                    prev[i][j] = max_idx
                else:
                    dp[i][j] = 2
                    prev[i][j] = -1
            else:
                # No valid h, sequence starts here
                dp[i][j] = 2
                prev[i][j] = -1
                
            if dp[i][j] > best_len:
                best_len = dp[i][j]
                best_i = i
                best_j = j
                
    # If we broke out of loops early due to time, we might have a partial result.
    # But best_len is tracked.
    # If best_len is 0, it means we didn't even process i=0?
    # If N>=2, i=0, j=1 will set best_len to 2.
    
    if best_len == 0:
        # Fallback
        return primes[:2] if N >= 2 else primes
        
    # 3. Reconstruction
    seq = [primes[best_j], primes[best_i]]
    i = best_i
    j = best_j
    while prev[i][j] != -1:
        h = prev[i][j]
        seq.append(primes[h])
        j = i
        i = h
        
    seq.reverse()
    return seq
