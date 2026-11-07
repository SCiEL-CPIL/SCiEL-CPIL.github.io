import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer
    
    # Generate blocks (p, k_block, T)
    # T = p^a * sigma(p^a)
    # k_block = p^a
    
    blocks = []
    
    # Primes up to 200 is sufficient for many small T
    # We can go higher if T is small
    # p^2 * (1+p) approx p^3 <= 10^18 => p <= 10^6
    # But we only need primes that can form small T.
    # Let's generate primes up to 1000 to be safe, but limit exponents.
    
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    # Generate primes up to 1000
    primes = []
    for i in range(2, 1000):
        if is_prime(i):
            primes.append(i)
            
    for p in primes:
        p_pow = p
        sigma_val = 1 + p
        while True:
            T = p_pow * sigma_val
            if T > 10**18:
                break
            blocks.append((p, p_pow, T))
            # Next exponent
            p_pow *= p
            sigma_val = sigma_val * p + 1
            if p_pow > 10**18:
                break
            
    # Sort blocks by T to allow more combinations to stay under limit
    blocks.sort(key=lambda x: x[2])
    
    M_to_ks = defaultdict(list)
    M_to_ks[1].append(1)
    
    max_count = 1
    best_M = 1
    
    # Iterative expansion
    for p, k_block, T in blocks:
        # Snapshot of current solutions
        # Using list(M_to_ks.items()) is O(N)
        items = list(M_to_ks.items())
        
        for M, ks in items:
            # Check time
            if time.time() > deadline:
                break
                
            M_new = M * T
            if M_new > 10**18:
                continue
                
            # Generate new k's
            for k in ks:
                if k % p == 0:
                    continue
                k_new = k * k_block
                M_to_ks[M_new].append(k_new)
                
                # Update max count
                cnt = len(M_to_ks[M_new])
                if cnt > max_count:
                    max_count = cnt
                    best_M = M_new
                    
            if time.time() > deadline:
                break
                
        if time.time() > deadline:
            break
            
    if max_count == 1:
        return {'M': 1, 'k': [1]}
        
    # Return the best M and its k's
    k_list = list(set(M_to_ks[best_M]))
    k_list.sort()
    
    return {'M': int(best_M), 'k': k_list}
