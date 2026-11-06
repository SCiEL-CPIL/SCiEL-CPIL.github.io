import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 90s kill
    TARGET = 123457
    
    # Step 1: Sieve of Eratosthenes to get all primes up to TARGET
    is_prime = [True] * (TARGET + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(TARGET**0.5) + 1):
        if is_prime[i]:
            # Optimization: start from i*i
            for j in range(i*i, TARGET + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, TARGET + 1) if is_prime[i]]
    
    # Step 2: Compute the largest prime factor P(n) for each n in [1, TARGET]
    # P(1) = 1
    largest_pf = [1] * (TARGET + 1)
    
    # Process primes in increasing order: for each prime p, mark all multiples.
    # Since we go in increasing order, the last prime that divides n gives P(n).
    for p in primes:
        for m in range(p, TARGET + 1, p):
            largest_pf[m] = p
            
    # Check time
    if time.time() > deadline:
        return [1, TARGET - 1]
    
    # Step 3: Linear scan over all pairs (a, b) where a + b = TARGET
    # We want to minimize max(P(a), P(b))
    best_max = TARGET + 1  # Upper bound
    best_a = 1
    best_b = TARGET - 1
    
    # Optimization: We can iterate a from 1 to TARGET//2 and check (a, TARGET-a)
    # and (TARGET-a, a) is the same pair. Since max(P(a), P(b)) is symmetric,
    # we only need to check a in [1, TARGET//2] if we consider unordered pairs.
    # However, the problem asks for a, b. The value is the same.
    # Iterating full range is fine and simpler, but half range is slightly faster.
    # Let's stick to full range for simplicity and correctness, it's O(N).
    
    for a in range(1, TARGET):
        b = TARGET - a
        pa = largest_pf[a]
        pb = largest_pf[b]
        current_max = pa if pa > pb else pb
        if current_max < best_max:
            best_max = current_max
            best_a = a
            best_b = b
            # Early termination: if we reach 2, it's the theoretical minimum 
            # (only possible if one is 1 and other is power of 2, or both powers of 2 summing to odd? No, sum is odd.
            # If a=1, P(1)=1. b=123456. P(123456) > 2.
            # If a is power of 2, P(a)=2. b is odd. P(b) >= 3.
            # So min max is at least 3.
            if best_max == 3:
                # Check if we can break? 
                # Can max be 2? 
                # a+b=odd. One is even, one is odd.
                # If even one is 2-power, P=2.
                # Odd one must be 1 (P=1) for max to be 2.
                # If b=1, a=123456. P(123456) != 2.
                # If a=1, b=123456. P(123456) != 2.
                # So max cannot be 2.
                # Can max be 3?
                # We need P(a)<=3 and P(b)<=3.
                # a,b are of form 2^x * 3^y.
                # Check if such a,b exist.
                # If found, 3 is the minimum possible value > 2.
                break
    
    return [best_a, best_b]
