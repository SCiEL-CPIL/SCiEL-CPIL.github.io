# EVOLVE-BLOCK-START
def construct():
    from math import isqrt
    
    # 1. Generate primes up to 10000
    is_prime = [True] * 10001
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, isqrt(10000) + 1):
        if is_prime[i]:
            for j in range(i*i, 10001, i):
                is_prime[j] = False
                
    primes = [p for p in range(2, 10001) if is_prime[p]]
    N = len(primes)
    
    if N == 0:
        return []
    if N == 1:
        return primes

    # 2. Greedy approach
    # Start with the first prime
    seq = [primes[0]]
    curr_prime = primes[0]
    prev_gap = 0
    
    # Index to start searching for the next prime
    # We can use binary search for efficiency
    import bisect
    
    # We want to find the smallest prime p in primes such that p > curr_prime and p - curr_prime >= prev_gap
    # i.e., p >= curr_prime + prev_gap
    
    while True:
        target = curr_prime + prev_gap
        # Find the first prime >= target
        # bisect_left returns the index where target would be inserted to keep order
        idx = bisect.bisect_left(primes, target)
        
        if idx >= N:
            break
            
        next_prime = primes[idx]
        
        # Check if it's strictly greater than curr_prime (it will be if target > curr_prime, which is true if prev_gap > 0)
        # If prev_gap == 0, target == curr_prime, bisect_left returns index of curr_prime.
        # We need next prime > curr_prime.
        if next_prime == curr_prime:
             idx += 1
             if idx >= N:
                 break
             next_prime = primes[idx]
             
        seq.append(next_prime)
        curr_prime = next_prime
        prev_gap = curr_prime - seq[-2] if len(seq) > 1 else 0
        
    return seq
# EVOLVE-BLOCK-END
