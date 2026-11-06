"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    # Generate primes up to 1000
    def primes_up_to(n):
        sieve = np.ones(n+1, dtype=bool)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5)+1):
            if sieve[i]:
                sieve[i*i::i] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = primes_up_to(1000)
    n_primes = len(primes)
    L_max = 100000
    
    # counts[x] is the number of primes covering x
    counts = np.zeros(L_max + 2, dtype=np.int32)
    
    # Initialize with random residues
    residues = np.array([np.random.randint(p) for p in primes], dtype=np.int32)
    
    # Build initial counts
    for i, p in enumerate(primes):
        r = residues[i]
        if r == 0:
            start = p
        else:
            start = r
        if start <= L_max:
            indices = np.arange(start, L_max + 1, p)
            counts[indices] += 1
            
    # Precompute indices for fast updates
    # Cache: (prime_index, residue) -> array of indices
    indices_cache = {}
    for i, p in enumerate(primes):
        for r in range(p):
            if r == 0:
                start = p
            else:
                start = r
            if start <= L_max:
                indices_cache[(i, r)] = np.arange(start, L_max + 1, p)
            else:
                indices_cache[(i, r)] = np.array([], dtype=np.int32)
                
    def get_L(counts):
        # Find largest L such that counts[1..L] >= 2
        # Check x=1 to L_max
        mask = counts[1:L_max+1] < 2
        if not np.any(mask):
            return L_max
        first_bad = np.argmax(mask)
        return first_bad
        
    current_L = get_L(counts)
    best_L = current_L
    best_residues = residues.copy()
    
    start_time = time.time()
    # Use 115s to be safe against 120s limit
    deadline = start_time + 115.0
    
    # Local Search
    while time.time() < deadline:
        # Pick a random prime to change
        i = np.random.randint(n_primes)
        p = primes[i]
        old_r = residues[i]
        
        # Pick a new random residue
        new_r = np.random.randint(p)
        if new_r == old_r:
            continue
            
        old_indices = indices_cache[(i, old_r)]
        new_indices = indices_cache[(i, new_r)]
        
        # Apply change
        counts[old_indices] -= 1
        counts[new_indices] += 1
        
        new_L = get_L(counts)
        
        if new_L > best_L:
            best_L = new_L
            best_residues = residues.copy()
            residues[i] = new_r
            current_L = new_L
        elif new_L == best_L:
            # Accept to explore plateau
            if np.random.rand() < 0.5:
                residues[i] = new_r
                current_L = new_L
            else:
                # Revert
                counts[old_indices] += 1
                counts[new_indices] -= 1
        else:
            # Revert
            counts[old_indices] += 1
            counts[new_indices] -= 1

    return list(best_residues)
# EVOLVE-BLOCK-END
