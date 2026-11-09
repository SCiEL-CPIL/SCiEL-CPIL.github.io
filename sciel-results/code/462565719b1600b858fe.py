import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s buffer for 90s limit
    
    # We need to check n up to 1000000, so n+1 up to 1000001
    LIMIT = 1000001
    
    # 1. Compute Smallest Prime Factors (SPF) using sieve
    spf = list(range(LIMIT + 1))
    for i in range(2, int(LIMIT**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Compute Powerful Part P(n) for all n using sieve-based approach
    # P(n) = product of p^e where e >= 2 in factorization of n
    # Strategy: For each prime p, for each power p^k with k >= 2,
    # mark all multiples of p^k to contribute p^k to their powerful part.
    
    P = [1] * (LIMIT + 1)
    
    for p in range(2, LIMIT + 1):
        if spf[p] != p:  # not prime
            continue
        
        # For this prime p, consider powers p^2, p^3, ...
        power = p * p
        while power <= LIMIT:
            # All multiples of 'power' have at least p^2 in their factorization
            # But we need the FULL p^e, so we need to be careful.
            # Actually, for a number n, if p^e || n (exactly e), then e >= 2,
            # we want to multiply P[n] by p^e.
            # 
            # Better approach: For each prime p, for each k >= 2,
            # find all n where p^k || n (exactly k), and multiply P[n] by p^k.
            # But that's complex. Alternative:
            # For each prime p, for each k >= 2, mark multiples of p^k.
            # Then for each such multiple, we know p^k divides n, but we need the exact exponent.
            #
            # Let's use a different approach:
            # For each prime p, compute for each n the exponent of p in n.
            # Then if exponent >= 2, multiply P[n] by p^exponent.
            #
            # Even better: Use the SPF to factor each number, but do it more efficiently.
            # Since we already have SPF, let's use a DP approach:
            # P(n) can be computed from P(n // spf[n]) if we track exponents.
            #
            # Actually, the simplest efficient approach:
            # For each prime p, for each k >= 2, let pk = p^k.
            # For each multiple m of pk, if (m // pk) is NOT divisible by p,
            # then the exponent of p in m is exactly k, so multiply P[m] by pk.
            # Otherwise, the exponent is higher, handled by larger k.
            #
            # This is O(N log N) total.
            
            # Find multiples where p^k divides n but p^(k+1) does not
            # i.e., n is divisible by p^k but not by p^(k+1)
            next_power = power * p
            
            for m in range(power, LIMIT + 1, power):
                if m % next_power != 0:
                    # p^k exactly divides m
                    P[m] *= power
            
            power = next_power
    
    # 3. Find max P(n)*P(n+1)
    max_val = 0
    best_n = 10000
    # n ranges from 10000 to 1000000
    for n in range(10000, 1000001):
        score = P[n] * P[n+1]
        if score > max_val:
            max_val = score
            best_n = n
            
    return best_n
