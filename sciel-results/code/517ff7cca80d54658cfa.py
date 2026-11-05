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
    
    points = list(range(N + 1))
    
    # Precompute primes up to N+1
    primes = _primes(N + 1)
    
    # Process primes in decreasing order.
    # Satisfying constraints for larger primes is "cheaper" (removes fewer elements)
    # because the minimum residue class size is smaller.
    # Since removing elements preserves the "missed class" property for subsets,
    # we can process all primes p <= initial_len in descending order.
    # If p > current_len, we skip it (it's not required).
    
    for p in reversed(primes):
        if time.time() > deadline:
            break
            
        if p > len(points):
            continue
            
        # Count residues efficiently
        counts = [0] * p
        for v in points:
            counts[v % p] += 1
            
        # Find the least populated residue class
        min_count = min(counts)
        min_residue = counts.index(min_count)
        
        # Remove all points in that residue class
        points = [v for v in points if v % p != min_residue]
        
    return points
