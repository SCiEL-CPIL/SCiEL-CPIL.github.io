import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin

    # Sieve of Eratosthenes to get primes up to 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    primes = [p for p in range(2, limit + 1) if is_prime[p]]
    n = len(primes)
    
    # Map prime value to index
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Max possible gap between consecutive primes <= 10000
    # Actually, the max gap between any two primes in our sequence could be up to ~10000
    # But we only need to track gaps up to the max consecutive prime gap
    # Let's compute max gap between consecutive primes
    max_consec_gap = max(primes[i+1] - primes[i] for i in range(n-1))
    # But gaps in our subsequence can be larger (skipping primes)
    # The max possible gap is primes[-1] - primes[0] = ~10000
    # But we can cap it: if gap > max_consec_gap, it's still valid
    # Actually, we need to track all possible gaps. Let's use a dict or array.
    # Max gap is at most limit - 2 = 9998
    # That's too large for an array. Let's use the fact that we only need gaps that actually occur.
    
    # Alternative: use the observation that for fixed i and new_gap, j is determined.
    # new_gap = primes[i] - primes[j], so primes[j] = primes[i] - new_gap
    # For this to be valid, primes[i] - new_gap must be a prime in our list.
    # So for each i, we iterate over j < i, compute new_gap, and update.
    # But that's O(n²) which is ~1.5M, fine.
    
    # Let's use: dp[i] = dict mapping last_gap -> (max_length, prev_j, prev_gap)
    # For efficiency, for each i, we want for each j < i:
    #   new_gap = primes[i] - primes[j]
    #   best_prev = max over g <= new_gap of dp[j][g]
    #   dp[i][new_gap] = max(dp[i][new_gap], best_prev + 1)
    
    # To compute "best_prev = max over g <= new_gap of dp[j][g]" quickly,
    # we can maintain for each j a sorted structure or just iterate since gaps are small.
    
    # Actually, let's just do O(n²) with a small constant.
    # For each j, dp[j] is a dict. The number of entries in dp[j] is at most the number of distinct gaps ending at j.
    # In practice, this should be small.
    
    dp = [dict() for _ in range(n)]
    # dp[i][g] = (length, prev_j, prev_g) for subsequence ending at prime i with last gap g
    
    # Initialize: each prime alone
    best_len = 1
    best_i = 0
    best_g = 0
    
    for i in range(n):
        if time.time() > deadline:
            break
        # Single element
        dp[i][0] = (1, -1, -1)
        
        for j in range(i):
            new_gap = primes[i] - primes[j]
            # Find best dp[j][g] for g <= new_gap
            best_val = -1
            best_g_prev = -1
            for g, (L, pj, pg) in dp[j].items():
                if g <= new_gap and L > best_val:
                    best_val = L
                    best_g_prev = g
            if best_val > 0:
                new_len = best_val + 1
                if new_gap not in dp[i] or new_len > dp[i][new_gap][0]:
                    dp[i][new_gap] = (new_len, j, best_g_prev)
                if new_len > best_len:
                    best_len = new_len
                    best_i = i
                    best_g = new_gap
    
    # Reconstruct
    seq = []
    i = best_i
    g = best_g
    while i != -1:
        L, pj, pg = dp[i][g]
        seq.append(primes[i])
        i = pj
        g = pg
    seq.reverse()
    return seq
