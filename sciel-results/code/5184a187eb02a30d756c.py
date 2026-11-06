import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 1000024
    n_min = 1000
    n_max = 1000000
    
    # Compute small_prime_part for m in 1..N
    # small_prime_part[m] = product of p^v_p(m) for primes p < 24
    small_part = list(range(N + 1))
    
    # Sieve of Eratosthenes
    sieve = bytearray(b'\x01') * (N + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(N**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    
    # For each prime p >= 27, remove all factors of p from multiples
    for p in range(27, N + 1):
        if sieve[p]:
            for k in range(p, N + 1, p):
                while small_part[k] % p == 0:
                    small_part[k] //= p
    
    # Sliding window: find n minimizing distinct small_prime_part values in [n+1, n+24]
    freq = defaultdict(int)
    distinct = 0
    best_n = n_min
    best_count = float('inf')
    
    # Initialize window for n = n_min
    for i in range(n_min + 1, n_min + 25):
        val = small_part[i]
        if freq[val] == 0:
            distinct += 1
        freq[val] += 1
    
    if distinct < best_count:
        best_count = distinct
        best_n = n_min
    
    # Slide window
    for n in range(n_min + 1, n_max + 1):
        if time.time() > deadline:
            break
        # Remove small_part[n] (was in window for n-1)
        val_remove = small_part[n]
        freq[val_remove] -= 1
        if freq[val_remove] == 0:
            distinct -= 1
        
        # Add small_part[n+24] (new in window for n)
        val_add = small_part[n + 24]
        if freq[val_add] == 0:
            distinct += 1
        freq[val_add] += 1
        
        if distinct < best_count:
            best_count = distinct
            best_n = n
    
    return best_n
