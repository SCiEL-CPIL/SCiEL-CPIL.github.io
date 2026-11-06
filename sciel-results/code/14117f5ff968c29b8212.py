"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000

# EVOLVE-BLOCK-START
import time
import numpy as np

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def _is_admissible(points, primes):
    for p in primes:
        residues = set(pt % p for pt in points)
        if len(residues) == p:
            return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit
    
    # Start from the full set [0, N]
    points = list(range(N + 1))
    
    # We need to check primes up to len(points)
    # As we remove points, len(points) decreases, so the required primes also decrease
    # But we can precompute primes up to N+1 initially
    
    all_primes = _primes(N + 1)
    
    # For efficiency, we'll iterate through primes and remove the least-populated residue class
    # when all p classes are covered
    
    changed = True
    while changed:
        changed = False
        current_len = len(points)
        # We only need to check primes <= current_len
        for p in all_primes:
            if p > current_len:
                break
            
            # Count residue classes
            residues = [pt % p for pt in points]
            counts = np.zeros(p, dtype=int)
            for r in residues:
                counts[r] += 1
            
            if counts.min() > 0:
                # All p residue classes are covered, need to remove one
                # Remove the least-populated class
                min_class = np.argmin(counts)
                points = [pt for pt in points if pt % p != min_class]
                changed = True
                break
        
        # Check time
        if time.time() > deadline:
            break
    
    # Final admissibility check - if not admissible, try to fix
    # But given our greedy approach, it should be admissible
    # However, let's verify and if needed, do a final repair pass
    final_len = len(points)
    primes_to_check = [p for p in all_primes if p <= final_len]
    
    for p in primes_to_check:
        residues = set(pt % p for pt in points)
        if len(residues) == p:
            # Not admissible, remove least-populated class
            counts = np.zeros(p, dtype=int)
            for pt in points:
                counts[pt % p] += 1
            min_class = np.argmin(counts)
            points = [pt for pt in points if pt % p != min_class]
    
    return points
# EVOLVE-BLOCK-END
