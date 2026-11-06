"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    target = 123457
    
    # Precompute Largest Prime Factor (LPF) for all numbers up to target
    # using a sieve for smallest prime factors (SPF).
    
    # Initialize SPF array
    spf = list(range(target + 1))
    
    # Sieve of Eratosthenes to find smallest prime factor
    for i in range(2, int(target**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, target + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute LPF for each number using iterative factorization
    # LPF[n] is the largest prime factor of n
    lpf = [1] * (target + 1)
    for i in range(2, target + 1):
        n = i
        max_pf = 1
        while n > 1:
            p = spf[n]
            if p > max_pf:
                max_pf = p
            while n % p == 0:
                n //= p
        lpf[i] = max_pf
    
    # Now search for a, b such that a + b = target
    # minimizing max(lpf[a], lpf[b])
    # Use a smarter approach: sort candidates by their LPF value
    # and find the pair with minimum max
    
    best_max_pf = float('inf')
    best_a = 1
    best_b = target - 1
    
    for a in range(1, target // 2 + 1):
        if time.time() > deadline:
            break
        
        b = target - a
        
        pa = lpf[a]
        pb = lpf[b]
        
        current_max = pa if pa > pb else pb
        
        if current_max < best_max_pf:
            best_max_pf = current_max
            best_a = a
            best_b = b
            
            # Early termination: since target is odd, one of a,b is even (P>=2)
            # and the other is odd (P>=3 if >1). So theoretical min is 3.
            if best_max_pf == 3:
                break

    return [best_a, best_b]
# EVOLVE-BLOCK-END
