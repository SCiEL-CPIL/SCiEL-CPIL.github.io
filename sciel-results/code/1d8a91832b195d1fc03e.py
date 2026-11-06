import time
import random

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    primes = [p for p in _primes_upto(1000) if p >= 13]
    n_primes = len(primes)
    
    L_MAX = 1000
    # We will expand L_MAX if we reach the cap
    
    residues = [(i + 1) % p for i, p in enumerate(primes)]
    
    count = [0] * (L_MAX + 1)
    for i, p in enumerate(primes):
        r = residues[i]
        start = r
        if start < 1: start = p
        for n in range(start, L_MAX + 1, p):
            count[n] += 1
            
    current_L = 0
    for n in range(1, L_MAX + 1):
        if count[n] > 0:
            current_L = n
        else:
            break
            
    best_L = current_L
    best_residues = residues[:]
    
    T = 1.0
    cooling = 0.9995
    
    while time.time() - start_time < time_limit:
        # Dynamic L_MAX expansion
        if current_L >= L_MAX - 10:
            # Extend
            old_max = L_MAX
            L_MAX *= 2
            count.extend([0] * (L_MAX - old_max))
            # Update counts for new range
            for i, p in enumerate(primes):
                r = residues[i]
                start = r
                if start < 1: start = p
                # Find first n > old_max
                # n = r + k*p > old_max
                k = (old_max - start) // p + 1
                if start + k*p < start: # overflow check, not needed here
                    pass
                first_n = start + k*p
                for n in range(first_n, L_MAX + 1, p):
                    count[n] += 1
            # Recalculate current_L
            for n in range(old_max + 1, L_MAX + 1):
                if count[n] > 0:
                    current_L = n
                else:
                    break
            
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        old_r = residues[i]
        new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            continue
            
        start_old = old_r
        if start_old < 1: start_old = p
        start_new = new_r
        if start_new < 1: start_new = p
        
        # Update counts
        for n in range(start_old, L_MAX + 1, p):
            count[n] -= 1
        for n in range(start_new, L_MAX + 1, p):
            count[n] += 1
            
        # Determine new L
        new_L = 0
        for n in range(1, L_MAX + 1):
            if count[n] > 0:
                new_L = n
            else:
                break
                
        if new_L > current_L:
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best_residues = residues[:]
            residues[i] = new_r
        else:
            delta = new_L - current_L
            if delta < 0:
                prob = pow(2.71828, delta / T)
                if random.random() > prob:
                    # Reject
                    for n in range(start_old, L_MAX + 1, p):
                        count[n] += 1
                    for n in range(start_new, L_MAX + 1, p):
                        count[n] -= 1
                else:
                    current_L = new_L
                    residues[i] = new_r
            else:
                current_L = new_L
                residues[i] = new_r
                
        T *= cooling
        if T < 0.01:
            T = 0.01

    return [[p, r] for p, r in zip(primes, best_residues)]
