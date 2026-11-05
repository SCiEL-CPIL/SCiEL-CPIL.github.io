"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import random
import time

def construct():
    # Generate primes up to 1000
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
    
    # We aim for L significantly higher than 1000 to be safe.
    # Let's check up to 5000.
    MAX_L = 5000
    
    # Initialize residues with 1
    residues = [1] * n_primes
    
    # Precompute indices for each prime
    # indices[i] = array of indices in 1..MAX_L divisible by primes[i]
    # Actually, we need indices congruent to r mod p.
    # We can generate these on the fly or precompute masks.
    # Precomputing masks for all p and all r is too much memory (168 * 1000 * 5000 bits? No, 168*1000 ints is fine, but 5000 length vectors is 168*1000*5000 = 840M entries, too big).
    # We will generate the slice indices on the fly.
    
    def get_indices(p, r, MAX_L):
        # Returns indices in 1..MAX_L such that index % p == r
        # If r == 0, indices are p, 2p, ...
        # If r != 0, indices are r, r+p, ... (if r < p)
        # Note: x is in 1..MAX_L.
        if r == 0:
            start = p
        else:
            start = r
            if start < 1: # r is 0..p-1, so r>=0. If r=0 handled above.
                start = p # This case doesn't happen for r>0
        if start > MAX_L:
            return np.array([], dtype=np.int32)
        return np.arange(start, MAX_L + 1, p, dtype=np.int32)

    # Initialize counts
    counts = np.zeros(MAX_L + 1, dtype=np.int32)
    
    def update_counts_add(counts, p, r):
        idxs = get_indices(p, r, MAX_L)
        if len(idxs) > 0:
            counts[idxs] += 1

    def update_counts_sub(counts, p, r):
        idxs = get_indices(p, r, MAX_L)
        if len(idxs) > 0:
            counts[idxs] -= 1

    # Initial setup
    for i, p in enumerate(primes):
        update_counts_add(counts, p, residues[i])
        
    def get_L(counts):
        # Find first x in 1..MAX_L with counts[x] < 2
        # counts[0] is ignored
        bad = np.where(counts[1:] < 2)[0]
        if len(bad) == 0:
            return MAX_L
        return bad[0]

    current_L = get_L(counts)
    best_L = current_L
    best_residues = residues[:]
    
    start_time = time.time()
    time_limit = 110.0
    
    # Greedy Local Search
    # We will do multiple passes. In each pass, we iterate through primes (shuffled)
    # and for each prime, we find the best residue.
    
    for pass_num in range(20): # Max 20 passes
        if time.time() - start_time > time_limit:
            break
            
        # Shuffle prime indices for diversity
        prime_indices = list(range(n_primes))
        random.shuffle(prime_indices)
        
        improved = False
        
        for idx in prime_indices:
            p = primes[idx]
            old_r = residues[idx]
            
            # Remove old residue
            update_counts_sub(counts, p, old_r)
            
            best_r_for_p = old_r
            best_L_for_p = -1
            
            # Try all residues for this prime
            # Optimization: We can sample residues if p is large? 
            # p <= 1000, so max 1000 residues. 168 primes * 1000 residues = 168k evals per pass.
            # Each eval involves:
            # 1. Add new r (update counts)
            # 2. Get L (argmax on 5000 array)
            # 3. Sub new r (update counts)
            # This is ~3 numpy ops per residue.
            # 168k * 3 ops = 500k numpy ops per pass.
            # Should be fast (< 1s per pass).
            
            for r in range(p):
                if r == old_r:
                    continue
                
                update_counts_add(counts, p, r)
                L_val = get_L(counts)
                update_counts_sub(counts, p, r)
                
                if L_val > best_L_for_p:
                    best_L_for_p = L_val
                    best_r_for_p = r
            
            # Apply best residue
            if best_r_for_p != old_r:
                residues[idx] = best_r_for_p
                update_counts_add(counts, p, best_r_for_p)
                improved = True
                
                # Update global best
                current_L = get_L(counts)
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues[:]
            else:
                # Re-add old residue if it was the best (or if no change)
                # Wait, if best_r_for_p == old_r, we removed it and didn't add anything back?
                # We removed old_r at the start of the loop.
                # If best is old_r, we must add it back.
                update_counts_add(counts, p, old_r)
                
        if not improved:
            # No improvement in this pass? 
            # Maybe we got stuck. Try a random restart or perturbation?
            # For now, break if no improvement to save time, or continue with random perturbations.
            # Let's do a few random perturbations if stuck.
            break

    return best_residues

# EVOLVE-BLOCK-END
