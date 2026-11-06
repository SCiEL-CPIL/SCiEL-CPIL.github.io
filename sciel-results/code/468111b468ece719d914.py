"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random
import math

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N)
X = np.arange(N + 1)
MODS = {p: X % p for p in PRIMES}

def construct():
    # Initialize residues randomly
    residues = {p: random.randint(0, p - 1) for p in PRIMES}
    
    def get_size(res):
        valid = np.ones(N + 1, dtype=bool)
        for p in PRIMES:
            valid &= (MODS[p] != res[p])
        return np.sum(valid)

    current_size = get_size(residues)
    best_size = current_size
    best_residues = residues.copy()
    
    # Simulated Annealing
    T = 100.0
    T_min = 1.0
    cooling = 0.995
    
    # Run for a fixed number of iterations or until temperature is low
    for i in range(5000):
        if T < T_min:
            break
        
        p = random.choice(PRIMES)
        old_r = residues[p]
        new_r = random.randint(0, p - 1)
        if new_r == old_r:
            continue
        
        residues[p] = new_r
        new_size = get_size(residues)
        
        delta = new_size - current_size
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current_size = new_size
            if current_size > best_size:
                best_size = current_size
                best_residues = residues.copy()
        else:
            residues[p] = old_r
            
        T *= cooling

    # Reconstruct the set from best_residues
    valid = np.ones(N + 1, dtype=bool)
    for p in PRIMES:
        valid &= (MODS[p] != best_residues[p])
        
    return [int(x) for x in X[valid]]
# EVOLVE-BLOCK-END
