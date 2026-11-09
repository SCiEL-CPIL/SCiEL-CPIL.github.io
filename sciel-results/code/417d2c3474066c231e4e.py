"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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

def _compute_L(counts, max_check=100000):
    """Find largest L such that every integer 1..L has count >= 2."""
    # counts is 1-indexed: counts[x] for x in 1..max_check
    L = 0
    for x in range(1, max_check + 1):
        if counts[x] >= 2:
            L = x
        else:
            break
    return L

def construct():
    primes = _get_primes(1000)
    n_primes = len(primes)  # 168
    max_check = 100000
    
    # Initialize with seed-like construction: prime i covers ceil(i/2)
    residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    
    # Build initial counts
    counts = np.zeros(max_check + 2, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        # Mark all x in 1..max_check where x ≡ r (mod p)
        start = r
        if start < 1:
            start += p * ((1 - start + p - 1) // p)  # first x >= 1 with x ≡ r mod p
        # Actually simpler: x = r, r+p, r+2p, ... but x must be >= 1
        # x ≡ r (mod p) means x = r + k*p for integer k
        # x >= 1: k >= ceil((1 - r) / p)
        if r == 0:
            first = p  # x = p, 2p, ... (since x=0 not in range)
        else:
            first = r
        for x in range(first, max_check + 1, p):
            counts[x] += 1
    
    current_L = _compute_L(counts, max_check)
    
    # Local search: try to improve by changing one residue at a time
    improved = True
    iterations = 0
    max_iterations = 5000
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        for i, p in enumerate(primes):
            old_r = residues[i]
            
            # Try all possible new residues
            best_new_r = old_r
            best_L = current_L
            
            for new_r in range(p):
                if new_r == old_r:
                    continue
                
                # Temporarily change residue
                # Decrement counts for old_r
                if old_r == 0:
                    first_old = p
                else:
                    first_old = old_r
                for x in range(first_old, max_check + 1, p):
                    counts[x] -= 1
                
                # Increment counts for new_r
                if new_r == 0:
                    first_new = p
                else:
                    first_new = new_r
                for x in range(first_new, max_check + 1, p):
                    counts[x] += 1
                
                # Compute new L
                new_L = _compute_L(counts, max_check)
                
                if new_L > best_L:
                    best_L = new_L
                    best_new_r = new_r
                
                # Revert: decrement new_r, increment old_r
                for x in range(first_new, max_check + 1, p):
                    counts[x] -= 1
                for x in range(first_old, max_check + 1, p):
                    counts[x] += 1
            
            if best_new_r != old_r:
                # Commit the change
                old_r = residues[i]
                residues[i] = best_new_r
                
                # Update counts
                if old_r == 0:
                    first_old = p
                else:
                    first_old = old_r
                for x in range(first_old, max_check + 1, p):
                    counts[x] -= 1
                
                if best_new_r == 0:
                    first_new = p
                else:
                    first_new = best_new_r
                for x in range(first_new, max_check + 1, p):
                    counts[x] += 1
                
                current_L = best_L
                improved = True
    
    return residues
# EVOLVE-BLOCK-END
