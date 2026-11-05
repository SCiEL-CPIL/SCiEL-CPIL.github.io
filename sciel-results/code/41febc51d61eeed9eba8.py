import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Precompute primes up to a reasonable limit
    # We need primes up to max p we test
    # Let's test p up to 10^6 initially
    limit = 1_000_000
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    
    primes = [i for i, is_prime in enumerate(sieve) if is_prime]
    
    best_k = 0
    best_p = 0
    K = 200
    
    for idx, p in enumerate(primes):
        if time.time() > deadline:
            break
            
        rem = [p*p + i + 1 for i in range(K)]
        
        # Sieve with primes q <= p
        # We use primes[0..idx]
        for q in primes[:idx+1]:
            # Check if q divides any rem[i]
            # And divide
            # Optimization: if q*q > max(rem), we can break? 
            # No, we need to remove factors up to p.
            # But if q > sqrt(max(rem)), then q can divide at most one rem[i], 
            # and that rem[i] must be q itself (if prime) or multiple.
            # Actually, if we just iterate q up to p, it's fine.
            
            # Find multiples
            start_val = p*p + 1
            end_val = p*p + K
            first = ((start_val + q - 1) // q) * q
            
            if first > end_val:
                continue
                
            for j in range(first, end_val + 1, q):
                i = j - (p*p + 1)
                while rem[i] % q == 0:
                    rem[i] //= q
                    
        # Check
        current_k = 0
        for i in range(K):
            if rem[i] == 1:
                current_k += 1
            else:
                break
                
        if current_k > best_k:
            best_k = current_k
            best_p = p
            
    return {'p': best_p, 'k': best_k}
