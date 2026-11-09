import time
import math

N = 5000

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
    deadline = start_time + 290  # Safety margin
    
    # Initialize with full set
    S = list(range(N + 1))
    
    # Precompute primes up to N
    primes_N = _primes(N)
    
    # We need to check primes p <= len(S).
    # len(S) decreases, so the set of relevant primes shrinks.
    
    while time.time() < deadline:
        # Check admissibility
        k = len(S)
        if k == 0:
            break
            
        # Find primes p <= k
        # We can filter primes_N
        relevant_primes = [p for p in primes_N if p <= k]
        
        violating_prime = None
        min_count_for_violating = float('inf')
        target_residue = None
        
        # To be efficient, we can check primes in increasing order?
        # Or find the one requiring smallest removal?
        # Let's check all relevant primes and find the one with the smallest "cost" to fix.
        # Cost = min count of any residue class.
        
        for p in relevant_primes:
            # Count residues
            counts = [0] * p
            for v in S:
                counts[v % p] += 1
            
            # Check if any is 0
            min_c = min(counts)
            if min_c > 0:
                # Violating
                if min_c < min_count_for_violating:
                    min_count_for_violating = min_c
                    violating_prime = p
                    target_residue = counts.index(min_c)
        
        if violating_prime is None:
            # Admissible
            break
        
        # Remove elements in target_residue mod violating_prime
        p = violating_prime
        r = target_residue
        S = [v for v in S if v % p != r]
        
    return S
