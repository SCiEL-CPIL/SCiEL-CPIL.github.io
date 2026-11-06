import math
import time
import random
import numpy as np

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    limit_time = start_time + 290.0
    
    primes = _primes(N)
    x = np.arange(N + 1)
    
    best_global_set = []
    best_global_size = 0
    
    # Try a few different initial K values
    initial_ks = [400, 500, 600, 700]
    
    for initial_k in initial_ks:
        if time.time() > limit_time:
            break
            
        K = initial_k
        residues = {}
        converged = False
        
        while not converged and time.time() < limit_time:
            # Update set of primes P
            P = [p for p in primes if p <= K]
            if not P:
                break
            
            # Precompute x % p for efficiency
            x_mod_p = {p: x % p for p in P}
            
            # Ensure residues dict matches P
            for p in P:
                if p not in residues:
                    residues[p] = random.randint(p)
            for p in list(residues.keys()):
                if p not in P:
                    del residues[p]
            
            # Helper to calculate count using precomputed mods
            def calc_count():
                mask = np.ones(N + 1, dtype=bool)
                for p in P:
                    r = residues[p]
                    mask[x_mod_p[p] == r] = False
                return mask.sum()
            
            count = calc_count()
            
            improved = True
            while improved and time.time() < limit_time:
                improved = False
                # Randomize order of primes to avoid bias
                for p in random.sample(P, len(P)):
                    old_r = residues[p]
                    best_r = old_r
                    best_c = count
                    for r in range(p):
                        if r == old_r:
                            continue
                        residues[p] = r
                        c = calc_count()
                        if c > best_c:
                            best_c = c
                            best_r = r
                    residues[p] = old_r
                    if best_r != old_r:
                        residues[p] = best_r
                        count = best_c
                        improved = True
                        
            K_new = count
            if K_new == K:
                converged = True
                # Update global best
                if count > best_global_size:
                    best_global_size = count
                    mask = np.ones(N + 1, dtype=bool)
                    for p in P:
                        mask[x_mod_p[p] == residues[p]] = False
                    best_global_set = [int(i) for i in np.where(mask)[0]]
            else:
                K = K_new
                # If K changed, P might change.
                # We restart the inner loop with updated P.
                # Note: x_mod_p needs to be recalculated if P changes.
                # This happens at the start of the while loop.
                
    return sorted(best_global_set)
# EVOLVE-BLOCK-END
