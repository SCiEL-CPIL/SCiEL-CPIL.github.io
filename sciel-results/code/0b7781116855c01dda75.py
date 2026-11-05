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
    # to satisfy constraints. The key insight from the knowledge:
    # "Greedy removal from full set [0,5000] in prime-order preserves far more
    # elements than the seed's small-start repair"
    #
    # For each prime p (in order), we need to ensure that not all p residue
    # classes are represented. We do this by removing all elements in the
    # least-populated residue class for that prime.
    #
    # After removals, the tuple size changes, which may change which primes
    # need to be checked. We iterate until stable.
    
    points = list(range(N + 1))
    n_points = N + 1
    
    # We need to check primes p <= len(points). As we remove points, len decreases.
    # Precompute primes up to N+1 (the max possible tuple size).
    primes = _primes(N + 1)
    
    # For efficiency, track which primes have been "satisfied" (i.e., have a missed residue class)
    # A prime p is satisfied if in the current set, not all residues 0..p-1 appear.
    
    # We'll iterate: for each prime p <= current len, check if all residues are present.
    # If yes, remove the least-populated residue class.
    # Repeat until no prime needs fixing.
    
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        current_len = len(points)
        
        # Only need to check primes p <= current_len
        for p in primes:
            if p > current_len:
                break
            if time.time() > deadline:
                break
            
            # Count residue classes
            classes = {}
            for v in points:
                r = v % p
                if r not in classes:
                    classes[r] = []
                classes[r].append(v)
            
            if len(classes) == p:
                # All residues present - need to remove one class
                # Remove the least-populated class
                smallest_class = min(classes.values(), key=len)
                smallest_set = set(smallest_class)
                points = [v for v in points if v not in smallest_set]
                changed = True
                break
    
    return points
