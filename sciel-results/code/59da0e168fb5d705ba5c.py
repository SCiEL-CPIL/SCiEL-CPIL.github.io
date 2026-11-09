import numpy as np
import time

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
    
    np.random.seed(42)
    residues = np.array([np.random.randint(p) for p in primes])
    
    best_residues = residues.copy()
    
    MAX_EVAL_L = 5000 
    
    def evaluate(residues):
        counts = np.zeros(MAX_EVAL_L + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                start = p
            else:
                start = r
            if start <= MAX_EVAL_L:
                indices = np.arange(start, MAX_EVAL_L + 1, p)
                counts[indices] += 1
        valid = counts >= 2
        first_bad = np.where(valid[1:] == False)[0]
        if len(first_bad) == 0:
            return MAX_EVAL_L
        else:
            return first_bad[0]
            
    best_score = evaluate(best_residues)
    
    start_time = time.time()
    timeout = 110
    
    T = 1.0
    cooling = 0.9999
    
    while time.time() - start_time < timeout:
        i = np.random.randint(n_primes)
        p = primes[i]
        
        old_r = residues[i]
        new_r = np.random.randint(p)
        
        if old_r == new_r:
            continue
            
        residues[i] = new_r
        
        new_score = evaluate(residues)
        
        if new_score > best_score:
            best_score = new_score
            best_residues = residues.copy()
            # Optional: reheat
            # T = min(1.0, T * 1.05)
        else:
            delta = new_score - best_score
            # If delta is 0, accept
            # If delta < 0, accept with prob exp(delta/T)
            if np.random.rand() < np.exp(delta / T):
                pass
            else:
                residues[i] = old_r
                
        T *= cooling
        if T < 0.001:
            T = 0.001 # Prevent underflow issues or just keep going
            
    return list(best_residues)
