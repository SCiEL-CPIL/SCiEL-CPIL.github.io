import time
import numpy as np

N = 5000
DEADLINE = 290  # 10s margin

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N + 1)

def construct():
    start_time = time.time()
    points = list(range(N + 1))
    
    # Cache primes for quick access
    # We need primes <= len(points)
    
    while True:
        if time.time() - start_time > DEADLINE:
            break
            
        k = len(points)
        if k == 0:
            break
            
        # Find bad primes
        bad_primes_info = [] # (p, min_residue, min_count)
        
        # Only check primes <= k
        relevant_primes = [p for p in PRIMES if p <= k]
        
        for p in relevant_primes:
            # Count residues
            counts = [0] * p
            for v in points:
                counts[v % p] += 1
            
            # Check if any residue is 0
            min_count = min(counts)
            if min_count == 0:
                continue # Admissible w.r.t p
            
            # Find residue with min count
            min_res = counts.index(min_count)
            bad_primes_info.append((p, min_res, min_count))
            
        if not bad_primes_info:
            break # Admissible
            
        # Select removal strategy
        # Strategy 1: Remove the class with the absolute minimum count
        # Strategy 2: Remove the class of the smallest prime with min count
        # Let's try Strategy 1 first.
        
        # Sort by count, then by prime size (smaller primes have larger classes usually? No, 5000/p)
        # Smaller p -> larger class size.
        # We want to remove few elements.
        # So we want min count.
        
        bad_primes_info.sort(key=lambda x: (x[2], x[0]))
        
        p_star, r_star, _ = bad_primes_info[0]
        
        # Remove all points congruent to r_star mod p_star
        points = [v for v in points if v % p_star != r_star]
        
    return points
