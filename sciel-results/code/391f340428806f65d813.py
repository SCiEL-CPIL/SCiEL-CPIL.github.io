import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # 85 seconds to be safe against 90s limit
    
    L = 10000
    R = 1000000
    
    max_val = R + 1
    
    # Sieve to compute Smallest Prime Factor
    spf = list(range(max_val + 1))
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if spf[i] == i:
            for j in range(i * i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Precompute Largest Prime Factor for all numbers up to max_val
    # LPF[n] = largest prime factor of n
    lpf = [1] * (max_val + 1)
    for i in range(2, max_val + 1):
        if spf[i] == i:  # i is prime
            lpf[i] = i
        else:
            p = spf[i]
            m = i // p
            # LPF(i) = max(p, LPF(m)) but we need to be careful
            # Actually, LPF(i) = max(p, LPF(i//p)) only if we divide out all p's
            # Simpler: LPF(n) = max(spf of each prime power in factorization)
            # Since spf gives smallest, we need largest prime factor
            # Let's compute it differently: for each prime p, mark multiples
            pass
    
    # Better approach: compute LPF by iterating primes
    # For each prime p, for each multiple of p, LPF[multiple] = max(LPF[multiple], p)
    # But that's O(n log log n) which is fine for n=1e6
    
    lpf = [1] * (max_val + 1)
    for i in range(2, max_val + 1):
        if spf[i] == i:  # i is prime
            for j in range(i, max_val + 1, i):
                if i > lpf[j]:
                    lpf[j] = i
    
    # Now find n in [L, R] that minimizes max(lpf[n], lpf[n+1])
    best_n = L
    best_val = float('inf')
    
    # Use a chunked approach to be cache-friendly
    chunk_size = 10000
    for start in range(L, R + 1, chunk_size):
        if time.time() > deadline:
            break
        end = min(start + chunk_size, R + 1)
        for n in range(start, end):
            current_max = max(lpf[n], lpf[n + 1])
            if current_max < best_val:
                best_val = current_max
                best_n = n
                # Early termination: if best_val is very small, we can stop
                # The theoretical minimum is 2 (both n and n+1 are powers of 2), 
                # but that's impossible for consecutive integers > 1
                # Next best is 3, then 5, etc.
                if best_val <= 5:
                    return best_n
    
    return best_n
# EVOLVE-BLOCK-END
