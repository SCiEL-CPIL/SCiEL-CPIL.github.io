"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
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


def construct():
    start_time = time.time()
    # 300s limit, use 290s to be safe
    deadline = start_time + 290.0
    
    # Start with all odd numbers in [0, N]
    # This ensures admissibility for p=2 (misses 0 mod 2)
    S = set(range(1, N + 1, 2))
    
    # Precompute primes up to N
    all_primes = _primes(N)
    
    while True:
        if time.time() > deadline:
            break
            
        k = len(S)
        if k == 0:
            break
            
        # Get primes <= k
        # Optimization: all_primes is sorted. We can break early.
        primes_to_check = []
        for p in all_primes:
            if p <= k:
                primes_to_check.append(p)
            else:
                break
        
        violation_found = False
        for p in primes_to_check:
            # Count residues mod p
            counts = [0] * p
            for v in S:
                counts[v % p] += 1
            
            # Check if any residue is missing
            if 0 in counts:
                continue # This prime is satisfied
            
            # Violation: all residues covered.
            # We must remove one residue class.
            # Choose the class with the smallest count to minimize loss.
            min_count = min(counts)
            target_residue = counts.index(min_count)
            
            # Remove all elements in S that are == target_residue mod p
            # Using set comprehension is faster than iterating and removing
            S = set(v for v in S if v % p != target_residue)
            
            violation_found = True
            break # Restart loop with new S and new k
            
        if not violation_found:
            break
            
    return sorted(list(S))
# EVOLVE-BLOCK-END
