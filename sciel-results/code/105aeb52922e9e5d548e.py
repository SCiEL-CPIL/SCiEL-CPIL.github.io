import time
import math

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 300  # 300s limit, keep margin
    
    # Start with full set
    A = list(range(N + 1))
    
    # Get primes up to N+1
    primes = _primes(N + 1)
    
    # Index for primes
    p_idx = 0
    p = primes[0] # 2
    
    while p_idx < len(primes) and p <= len(A):
        if time.time() > deadline - 10:
            break
            
        # Check if all residues mod p are present in A
        residues = [0] * p
        for x in A:
            residues[x % p] = 1
            
        if all(residues):
            # Need to remove a residue class
            # Find the residue class with the fewest elements in A
            min_count = float('inf')
            best_r = -1
            # We can count more efficiently
            counts = [0] * p
            for x in A:
                counts[x % p] += 1
                
            for r in range(p):
                if counts[r] < min_count:
                    min_count = counts[r]
                    best_r = r
                    
            # Remove all x in A such that x % p == best_r
            A = [x for x in A if x % p != best_r]
            
        p_idx += 1
        if p_idx < len(primes):
            p = primes[p_idx]
            
    return A
