import time
import random
import numpy as np

def construct():
    # Primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    primes = sieve(1000)
    n_primes = len(primes)
    
    MAX_L = 10000 
    
    # Initialize residues randomly
    residues = np.array([random.randint(0, p - 1) for p in primes], dtype=int)
    
    def compute_counts(res):
        counts = np.zeros(MAX_L + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = res[i]
            if r == 0:
                idx = np.arange(p, MAX_L + 1, p)
            else:
                start = r
                if start < 1:
                    start += p
                idx = np.arange(start, MAX_L + 1, p)
            if len(idx) > 0:
                counts[idx] += 1
        return counts

    def find_first_gap(counts):
        bad_mask = counts[1:MAX_L+1] < 2
        if not np.any(bad_mask):
            return MAX_L + 1
        first_bad_idx = np.argmax(bad_mask)
        return first_bad_idx + 1

    counts = compute_counts(residues)
    current_L = find_first_gap(counts) - 1
    best_L = current_L
    best_residues = residues.copy()
    
    start_time = time.time()
    time_limit = 115.0
    
    T = 10.0
    cooling_rate = 0.995
    
    while time.time() - start_time < time_limit:
        if current_L >= MAX_L:
            break
            
        if random.random() < 0.8:
            gap = find_first_gap(counts)
            if gap > MAX_L:
                break
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            old_r = residues[i]
            
            if gap % p == old_r:
                # Already covers, so changing to same is no-op.
                # Change to random to explore?
                new_r = random.randint(0, p - 1)
                if new_r == old_r:
                    continue
            else:
                new_r = gap % p
        else:
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            old_r = residues[i]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue

        # Apply move
        if old_r == 0:
            idx_old = np.arange(p, MAX_L + 1, p)
        else:
            start_old = old_r
            if start_old < 1: start_old += p
            idx_old = np.arange(start_old, MAX_L + 1, p)
            
        if len(idx_old) > 0:
            counts[idx_old] -= 1
            
        if new_r == 0:
            idx_new = np.arange(p, MAX_L + 1, p)
        else:
            start_new = new_r
            if start_new < 1: start_new += p
            idx_new = np.arange(start_new, MAX_L + 1, p)
            
        if len(idx_new) > 0:
            counts[idx_new] += 1
            
        residues[i] = new_r
        
        new_L = find_first_gap(counts) - 1
        
        delta_L = new_L - current_L
        
        if delta_L >= 0 or random.random() < np.exp(delta_L / T):
            current_L = new_L
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
        else:
            if len(idx_old) > 0:
                counts[idx_old] += 1
            if len(idx_new) > 0:
                counts[idx_new] -= 1
            residues[i] = old_r
            
        T *= cooling_rate
        if T < 0.1:
            T = 0.1

    return best_residues.tolist()

construct()
