import time
import math

N = 5000
_DEADLINE = 320  # safety margin before 330s wall-clock kill

def _primes(limit):
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p*p::p] = bytearray(len(sieve[p*p::p]))
    return [p for p in range(2, limit + 1) if sieve[p]]

def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    # Greedy removal from full set [0, N]
    # We want the largest admissible tuple: for every prime p <= len(tuple),
    # some residue class mod p is missed.
    #
    # Strategy: Start with all integers [0, N]. Iteratively remove integers
    # to satisfy constraints.
    #
    # Key improvement: Process primes in DESCENDING order (largest first).
    # For large p, removing one residue class only removes ~1/p of the set,
    # which is a small fraction. For small p, removing one class removes ~1/p
    # which is a large fraction. By handling large primes first (while the set
    # is still large and they are active constraints), we minimize total removals.
    #
    # After removals, the tuple size changes, which may change which primes
    # need to be checked. We iterate until stable.
    
    points = list(range(N + 1))
    
    # We need to check primes p <= len(points). As we remove points, len decreases.
    # Precompute primes up to N+1 (the max possible tuple size).
    primes = _primes(N + 1)
    # Sort in descending order to process largest primes first
    primes_desc = primes[::-1]
    
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        current_len = len(points)
        
        # Only need to check primes p <= current_len, in descending order
        for p in primes_desc:
            if p > current_len:
                continue
            if time.time() > deadline:
                break
            
            # Count residue classes using an array for speed
            counts = [0] * p
            for v in points:
                counts[v % p] += 1
            
            # Check if all residues are present
            if min(counts) > 0:
                # All residues present - need to remove one class
                # Remove the least-populated class
                min_idx = counts.index(min(counts))
                points = [v for v in points if v % p != min_idx]
                changed = True
                break
    
    return points
