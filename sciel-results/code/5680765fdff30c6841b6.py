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
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin
    
    # Start with all integers
    points = list(range(N + 1))
    
    # List of primes up to N
    primes = _primes(N)
    
    # We iterate through primes in increasing order.
    # We only need to process primes p such that p <= len(points)
    # Since len(points) decreases, we can break early if p > len(points)
    
    for p in primes:
        if time.time() > deadline:
            break
            
        current_size = len(points)
        if p > current_size:
            break
            
        # Check if points covers all residues mod p
        # Count occurrences of each residue
        counts = [0] * p
        for v in points:
            counts[v % p] += 1
            
        # If any count is 0, it's admissible for this p
        if 0 in counts:
            continue
            
        # It covers all residues. We must remove one class.
        # Choose the class with the smallest count to minimize loss
        min_count = min(counts)
        # Find the residue with min_count
        # If multiple, pick the first one (arbitrary)
        target_residue = counts.index(min_count)
        
        # Remove all points congruent to target_residue mod p
        points = [v for v in points if v % p != target_residue]
        
    return points
