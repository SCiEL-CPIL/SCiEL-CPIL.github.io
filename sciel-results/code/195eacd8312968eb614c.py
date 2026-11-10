import time

def construct():
    start_time = time.time()
    deadline = 80  # safety margin before 90s kill
    limit = 1000000
    max_val = limit + 1
    
    # Initialize largest prime factor array
    lpf = [0] * (max_val + 1)
    
    # Sieve to find largest prime factor
    # Handle 2 separately
    for j in range(2, max_val + 1, 2):
        lpf[j] = 2
    
    # For odd primes only
    for p in range(3, max_val + 1, 2):
        if lpf[p] == 0:  # p is prime
            for j in range(p, max_val + 1, p):
                lpf[j] = p
    
    # Check time after sieve
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return 10000
    
    best_n = 10000
    min_score = float('inf')
    
    # Scan for the pair with minimum max LPF
    for n in range(10000, limit + 1):
        score = max(lpf[n], lpf[n+1])
        if score < min_score:
            min_score = score
            best_n = n
            # Early exit if we found the theoretical minimum
            # LPF must be at least 2, and for consecutive ints,
            # one is even so LPF >= 2. If we find score=2, that's optimal.
            if min_score <= 2:
                break
    
    return best_n
