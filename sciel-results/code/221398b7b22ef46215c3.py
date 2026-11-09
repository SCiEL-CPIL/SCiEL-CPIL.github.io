# EVOLVE-BLOCK-START
import numpy as np

def _get_primes(n=1000):
    """Return list of primes <= n in increasing order."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    primes = _get_primes(1000)
    n_primes = len(primes)  # 168
    max_check = 100000
    
    # Initialize residues with a simple pattern
    residues = [0] * n_primes
    
    # Build initial counts
    counts = np.zeros(max_check + 2, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        if r == 0:
            start = p
        else:
            start = r
        counts[start::p] += 1
        
    def get_L(c):
        bad = np.where(c[1:max_check+1] < 2)[0]
        if len(bad) == 0:
            return max_check
        return bad[0]
        
    current_L = get_L(counts)
    
    improved = True
    max_iterations = 200
    iterations = 0
    stuck_counter = 0
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        x_target = current_L + 1
        
        if x_target > max_check:
            break
            
        # Try to cover x_target
        for i, p in enumerate(primes):
            old_r = residues[i]
            
            if x_target % p == old_r:
                continue
                
            new_r = x_target % p
            
            if old_r == 0:
                start_old = p
            else:
                start_old = old_r
            counts[start_old::p] -= 1
            
            if new_r == 0:
                start_new = p
            else:
                start_new = new_r
            counts[start_new::p] += 1
            
            new_L = get_L(counts)
            
            if new_L > current_L:
                residues[i] = new_r
                current_L = new_L
                improved = True
                stuck_counter = 0
                break
            else:
                counts[start_new::p] -= 1
                counts[start_old::p] += 1
        
        if not improved:
            stuck_counter += 1
            if stuck_counter > 5:
                # Random kick
                for _ in range(3):
                    idx = np.random.randint(n_primes)
                    p = primes[idx]
                    old_r = residues[idx]
                    new_r = np.random.randint(p)
                    
                    if old_r == 0:
                        start_old = p
                    else:
                        start_old = old_r
                    counts[start_old::p] -= 1
                    
                    if new_r == 0:
                        start_new = p
                    else:
                        start_new = new_r
                    counts[start_new::p] += 1
                    
                    residues[idx] = new_r
                current_L = get_L(counts)
                stuck_counter = 0
                improved = True # Force another pass

    return residues
# EVOLVE-BLOCK-END
