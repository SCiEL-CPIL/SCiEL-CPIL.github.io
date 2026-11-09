import time
import numpy as np

N = 5000
DEADLINE = 300.0
START_TIME = time.time()

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    global START_TIME
    if time.time() - START_TIME > DEADLINE - 5:
        return list(range(0, N + 1))
    
    # Start with all integers in [0, N]
    points = list(range(N + 1))
    
    # Get all primes up to N (since tuple size <= N+1)
    all_primes = _primes(N + 1)
    
    # Process primes in order; for each prime p, if all residue classes mod p are present,
    # remove the smallest residue class
    # We need to iterate until stable
    changed = True
    while changed:
        if time.time() - START_TIME > DEADLINE - 5:
            break
        changed = False
        for p in all_primes:
            if p > len(points):
                break
            # Check if all p residue classes are covered
            residues = {}
            for v in points:
                r = v % p
                if r not in residues:
                    residues[r] = 0
                residues[r] += 1
            if len(residues) == p:
                # All classes covered; remove the smallest class
                min_class = min(residues, key=lambda r: residues[r])
                points = [v for v in points if v % p != min_class]
                changed = True
                break  # Restart the prime loop since points changed
    
    return points
