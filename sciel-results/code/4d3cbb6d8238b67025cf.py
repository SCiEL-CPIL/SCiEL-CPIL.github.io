"""Rewritten program for narrow admissible tuple in [0, 5000]."""
import time
import math
import numpy as np

N = 5000

# EVOLVE-BLOCK-START

def _primes(limit):
    """Generate all primes up to limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = bytearray(len(sieve[p * p::p]))
    return [i for i in range(limit + 1) if sieve[i]]


def construct():
    """Construct the largest admissible tuple in [0, N].
    
    An admissible tuple satisfies: for every prime p <= len(tuple),
    some residue class mod p is completely missed by the tuple.
    """
    start_time = time.time()
    TIME_LIMIT = 280  # Safety margin before 300s evaluator kill
    
    # Start with all integers in [0, N]
    active = np.ones(N + 1, dtype=bool)
    L = N + 1
    
    # Precompute primes up to N+1
    all_primes = _primes(N + 1)
    
    # Initialize residue counts for all primes <= N+1
    res_counts = {}
    for p in all_primes:
        if p > L:
            break
        # Use numpy for efficient counting
        residues = np.arange(N + 1) % p
        counts = np.bincount(residues, minlength=p)
        res_counts[p] = counts
        
    # Identify initially violated primes (all classes > 0)
    violated = set()
    for p in all_primes:
        if p > L:
            break
        if res_counts[p].min() > 0:
            violated.add(p)
            
    while True:
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT:
            break
        
        # Find the smallest prime p <= L that is still in violated
        target_p = None
        for p in all_primes:
            if p > L:
                break
            if p in violated:
                target_p = p
                break

        if target_p is None:
            break  # Admissible!
            
        p = target_p
        
        # Find the residue class with the minimum count
        counts = res_counts[p]
        min_count = counts.min()
        min_r = np.argmin(counts)
        
        # Remove all points v such that v % p == min_r and active[v]
        # Using numpy for efficiency
        candidates = np.arange(N + 1)[active & (np.arange(N + 1) % p == min_r)]
        
        if len(candidates) == 0:
            # No candidates to remove, should not happen but safety check
            violated.discard(p)
            continue
            
        for v in candidates:
            active[v] = False
            L -= 1
            
            # Update counts for all primes q <= L
            for q in all_primes:
                if q > L:
                    continue
                r = v % q
                res_counts[q][r] -= 1
                if res_counts[q][r] == 0:
                    violated.discard(q)
                    
        # p is now satisfied
        violated.discard(p)
        
    points = [int(v) for v in np.where(active)[0]]
    
    return points

# EVOLVE-BLOCK-END
