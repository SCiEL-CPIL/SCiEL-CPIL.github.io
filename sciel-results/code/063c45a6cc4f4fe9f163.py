import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s buffer before 90s kill
    
    N = 1000000
    MAX_N = N + 1  # we need n and n+1, so up to 1000001
    
    # powerful_part[i] will hold the powerful part of i
    powerful_part = [1] * (MAX_N + 1)
    
    # Sieve for smallest prime factor
    spf = list(range(MAX_N + 1))
    for i in range(2, int(MAX_N**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # For each prime p, for each power p^e with e >= 2,
    # multiply p^e into powerful_part[m] for all m divisible by p^e
    
    # First, collect all primes up to MAX_N
    primes = []
    for i in range(2, MAX_N + 1):
        if spf[i] == i:
            primes.append(i)
    
    # For each prime, iterate over its powers >= 2
    for p in primes:
        power = p * p  # p^2
        while power <= MAX_N:
            # For all multiples of power, multiply by power
            for m in range(power, MAX_N + 1, power):
                powerful_part[m] *= power
            power *= p
            if power > MAX_N:
                break
    
    # Check time
    if time.time() > deadline - 10:
        # Fallback: just find best among first chunk
        best_n = 10000
        best_val = 0
        for n in range(10000, 50000):
            val = powerful_part[n] * powerful_part[n+1]
            if val > best_val:
                best_val = val
                best_n = n
        return best_n
    
    # Find n in [10000, 1000000] that maximizes powerful_part[n] * powerful_part[n+1]
    best_n = 10000
    best_val = 0
    for n in range(10000, N + 1):
        val = powerful_part[n] * powerful_part[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
        if time.time() > deadline - 5:
            break
    
    return best_n
