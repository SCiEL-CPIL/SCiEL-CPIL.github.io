import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if sieve[i]:
                sieve[i*i:n+1:i] = [False] * len(sieve[i*i:n+1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    n_primes = len(primes)
    
    residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    
    L_limit = 5000
    counts = np.zeros(L_limit + 1, dtype=np.int32)
    
    def update_counts(res):
        nonlocal counts
        counts = np.zeros(L_limit + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = res[i]
            start = r if r > 0 else p
            if start <= L_limit:
                counts[start:L_limit+1:p] += 1

    update_counts(residues)
    
    def get_first_failure():
        bad = np.where(counts[1:] < 2)[0]
        if len(bad) == 0:
            return L_limit + 1
        return bad[0] + 1
        
    current_L = get_first_failure()
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        next_x = current_L
        
        if next_x > L_limit:
            L_limit = L_limit * 2
            update_counts(residues)
            current_L = get_first_failure()
            if current_L > L_limit // 2: # It extended
                improved = True
            continue
            
        # Find candidates
        best_candidate = -1
        
        for i, p in enumerate(primes):
            r_old = residues[i]
            r_new = next_x % p
            
            if r_old == r_new:
                continue
            
            start_old = r_old if r_old > 0 else p
            # Check if any x in [start_old, next_x, step p] has count == 2
            # Slice: counts[start_old:next_x:p]
            segment = counts[start_old:next_x:p]
            if len(segment) > 0 and np.any(segment == 2):
                continue
            else:
                best_candidate = i
                break
                
        if best_candidate != -1:
            i = best_candidate
            p = primes[i]
            r_old = residues[i]
            r_new = next_x % p
            
            # Update counts
            start_old = r_old if r_old > 0 else p
            counts[start_old:L_limit+1:p] -= 1
            
            start_new = r_new if r_new > 0 else p
            counts[start_new:L_limit+1:p] += 1
            
            residues[i] = r_new
            
            new_L = get_first_failure()
            if new_L > current_L:
                current_L = new_L
                improved = True

    return residues
