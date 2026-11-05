"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

N = 5000
DEADLINE = 320  # seconds, safety margin before 330s kill

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]


def _is_admissible(points, primes_cache):
    """Check if points is admissible: for every prime p <= len(points), some residue class mod p is missed."""
    n = len(points)
    for p in primes_cache:
        if p > n:
            break
        # Check if all p residue classes are covered
        covered = set()
        for v in points:
            covered.add(v % p)
            if len(covered) == p:
                return False
        # If we get here, not all classes covered -> OK for this prime
    return True


def construct():
    start_time = time.time()
    
    # Precompute primes up to N+1 (max possible length is N+1=5001)
    max_len = N + 1
    primes_list = _primes(max_len)
    
    # Greedy approach: start with all numbers, remove to satisfy admissibility
    points = list(range(N + 1))
    
    # Iteratively remove least-populated residue class for violating primes
    changed = True
    while changed:
        if time.time() - start_time > DEADLINE:
            break
        
        changed = False
        n = len(points)
        
        for p in primes_list:
            if p > n:
                break
            
            # Count residue classes
            classes = {}
            for v in points:
                r = v % p
                if r not in classes:
                    classes[r] = []
                classes[r].append(v)
            
            if len(classes) == p:
                # All residue classes covered - need to remove one
                # Remove the least-populated class
                smallest_class = min(classes.keys(), key=lambda r: len(classes[r]))
                to_remove = set(classes[smallest_class])
                points = [v for v in points if v not in to_remove]
                changed = True
                break
    
    return points
# EVOLVE-BLOCK-END
