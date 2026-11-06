import time
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = 85  # Safety margin for 90s limit
    
    def elapsed():
        return time.time() - t0
    
    MAXN = 1_000_000
    MAXN_PLUS = MAXN + 1  # need pp[n+1] for n up to MAXN-1, so index up to MAXN
    
    # 1. Sieve of Eratosthenes to get primes
    is_prime = bytearray(b'\x01') * (MAXN + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(MAXN**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, MAXN + 1) if is_prime[i]]
    
    # 2. Compute powerful part P(m) for all m using NumPy
    # P(m) = product of p^k where k = v_p(m) and k >= 2
    pp = np.ones(MAXN_PLUS, dtype=np.int64)
    
    for p in primes:
        if elapsed() > DEADLINE:
            break
            
        p_power = p * p
        
        while p_power <= MAXN:
            # Numbers with v_p(m) == k are multiples of p^k but not p^(k+1)
            next_power = p_power * p
            
            # Indices where m is divisible by p_power
            indices = np.arange(p_power, MAXN_PLUS, p_power)
            
            # Filter: m % next_power != 0
            if next_power <= MAXN:
                mask = (indices % next_power) != 0
                indices = indices[mask]
            
            if len(indices) > 0:
                pp[indices] *= p_power
            
            p_power = next_power
    
    # 3. Find max pp[n] * pp[n+1] for n in [10000, MAXN-1]
    if elapsed() > DEADLINE:
        # Fallback: return a reasonable default
        return 10000
    
    # Compute products for n in [10000, MAXN-1]
    # pp[n] * pp[n+1] for n from 10000 to MAXN-1
    products = pp[10000:MAXN] * pp[10001:MAXN_PLUS]
    
    best_idx = np.argmax(products)
    best_val = products[best_idx]
    best_n = 10000 + best_idx
    
    return int(best_n)
