"""Optimized program using vectorized NumPy sieve for powerful parts."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 75  # 75s safety margin (90s limit)
    
    N = 1000001  # We need powerful parts for 10000..1000000 and 100001..1000001
    
    # powerful_part array: for each i, store the product of p^e for all prime powers
    # p^e where e >= 2 in the factorization of i
    pw = np.ones(N + 1, dtype=np.int64)
    
    # Sieve for primes up to N
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    
    # Simple sieve to get primes
    for i in range(2, int(np.sqrt(N)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.where(is_prime)[0]
    
    # For each prime, for each power p^k with k >= 2, multiply pw at indices divisible by p^k
    # by the full prime power p^k (not just p^2, but the complete p^e where e is the actual exponent)
    # 
    # Strategy: for each prime p, find the maximum power p^max <= N
    # For each k from 2 to max, we need to handle indices where p^k divides i but p^(k+1) does not
    # (i.e., the exponent of p in i is exactly k). For those, multiply by p^k.
    # For indices where p^(k+1) divides i, they'll be handled at the higher power.
    #
    # Simpler approach: for each prime p, for each k >= 2 where p^k <= N:
    #   indices where p^k | i but p^(k+1) !| i get multiplied by p^k
    #   This is: (i % p^k == 0) AND (i % p^(k+1) != 0)
    
    for p in primes:
        p2 = p * p
        if p2 > N:
            break
        
        pk = p2
        k = 2
        while pk <= N:
            # Next power
            pk_next = pk * p
            if pk_next <= N:
                # Indices divisible by pk but NOT by pk_next
                # Use sparse approach: indices = pk, 2*pk, 3*pk, ... up to N
                # Then exclude those divisible by pk_next
                idx = np.arange(pk, N + 1, pk)  # all multiples of pk
                if len(idx) > 0:
                    # Exclude multiples of pk_next
                    mask = (idx % pk_next) != 0
                    idx_filtered = idx[mask]
                    if len(idx_filtered) > 0:
                        pw[idx_filtered] *= pk
            else:
                # pk is the highest power <= N, so all multiples of pk get pk
                idx = np.arange(pk, N + 1, pk)
                if len(idx) > 0:
                    pw[idx] *= pk
            
            # Check deadline
            if time.time() > deadline - 5:
                break
            
            pk = pk_next
            k += 1
        
        if time.time() > deadline - 5:
            break
    
    # Now compute the product of powerful parts of consecutive integers
    # For n in [10000, 1000000], product = pw[n] * pw[n+1]
    products = pw[10000:1000001] * pw[10001:1000002]
    
    # Find the index of the maximum product
    max_idx = np.argmax(products)
    best_n = 10000 + max_idx
    
    return int(best_n)

# EVOLVE-BLOCK-END
