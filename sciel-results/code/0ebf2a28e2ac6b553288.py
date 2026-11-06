import time

def construct():
    t0 = time.time()
    DEADLINE = 85  # Safety margin before 90s kill
    
    def elapsed():
        return time.time() - t0

    MAXN = 1_000_000
    # pp[i] will store the powerful part of i
    pp = [1] * (MAXN + 2)
    
    # Sieve for primes
    is_prime = bytearray(b'\x01') * (MAXN + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(MAXN**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, MAXN + 1) if is_prime[i]]
    
    # Efficient powerful part calculation
    # For each prime p, for each power p^k (k>=2), 
    # multiply pp[m] by p^k for all m such that v_p(m) == k.
    # v_p(m) == k iff p^k divides m and p^(k+1) does not divide m.
    
    for p in primes:
        p_k = p * p
        k = 2
        while p_k <= MAXN:
            # Next power p^(k+1)
            p_next = p_k * p
            # Step is p_k. We want multiples of p_k that are not multiples of p_next.
            # Multiples of p_k: p_k, 2*p_k, 3*p_k, ...
            # Among these, those not divisible by p_next are those where the multiplier is not divisible by p.
            # So we iterate m = j * p_k where j is not divisible by p.
            # j = 1, 2, ..., floor(MAXN / p_k)
            # Skip j if j % p == 0.
            
            limit = MAXN // p_k
            # Iterate j
            # To avoid modulo in inner loop, we can iterate in blocks or just use modulo.
            # Modulo is fast enough in Python for this count (~0.45 * MAXN ops total).
            for j in range(1, limit + 1):
                if j % p != 0:
                    m = j * p_k
                    pp[m] *= p_k
            
            if elapsed() > DEADLINE:
                # If we run out of time, we have a partial pp array.
                # We should probably break and search what we have, or return a safe default.
                # But the sieve is fast, this is unlikely to trigger.
                break
                
            p_k = p_next
            k += 1
            if p_k > MAXN:
                break
        
        if elapsed() > DEADLINE:
            break
    
    if elapsed() > DEADLINE:
        # Fallback if sieve didn't finish (should not happen)
        return 10000

    # Scan for the maximum product pp[n] * pp[n+1]
    best_val = 0
    best_n = 10000
    
    # We only need to check n where pp[n] > 1 or pp[n+1] > 1
    # But a simple loop is fast enough (1 million iterations)
    for n in range(10000, MAXN):
        if elapsed() > DEADLINE:
            break
        prod = pp[n] * pp[n + 1]
        if prod > best_val:
            best_val = prod
            best_n = n
            
    return best_n
