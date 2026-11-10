import numpy as np
import time
import random

def construct():
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    primes = sieve(1000)
    n_primes = len(primes)
    P = np.array(primes)
    
    # Initial residues: random
    residues = np.array([random.randint(0, p - 1) for p in P])
    
    def get_score(res):
        # Check x=1..L_max for first failure
        # Use chunked evaluation with efficient marking
        L_max = 10000
        chunk = 100
        x = 1
        while x <= L_max:
            end = min(x + chunk - 1, L_max)
            counts = np.zeros(end - x + 1, dtype=np.int32)
            
            for i in range(n_primes):
                p = P[i]
                r = res[i]
                
                # Find first hit in [x, end]
                # x % p == r
                # x = k*p + r
                if x <= r:
                    first_hit = r
                    if first_hit < 1:
                        first_hit += p
                else:
                    k = (x - r + p - 1) // p
                    first_hit = k * p + r
                    
                if first_hit <= end:
                    # Mark all hits
                    indices = np.arange(first_hit, end + 1, p)
                    rel_indices = indices - x
                    counts[rel_indices] += 1
            
            fail_indices = np.where(counts < 2)[0]
            if len(fail_indices) > 0:
                return x + fail_indices[0] - 1
            x = end + 1
            
        return L_max

    best_score = get_score(residues)
    best_res = residues.copy()
    
    start_time = time.time()
    time_limit = 140.0
    
    current_score = best_score
    current_res = best_res.copy()
    
    # Local search
    while time.time() - start_time < time_limit:
        # Random mutation
        idx = random.randint(0, n_primes - 1)
        p = P[idx]
        old_val = current_res[idx]
        new_val = random.randint(0, p - 1)
        if new_val == old_val:
            continue
            
        current_res[idx] = new_val
        new_score = get_score(current_res)
        
        if new_score >= current_score:
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_res = current_res.copy()
        else:
            # Revert
            current_res[idx] = old_val
            
    return best_res.tolist()

# EVOLVE-BLOCK-END
