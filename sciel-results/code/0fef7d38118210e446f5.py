"""Rewritten program for the narrow admissible tuple task (fixed-diameter maximization)."""
import time
import random
import numpy as np

N = 5000
DEADLINE = 320.0  # safety margin before 330s wall-clock kill

# Precompute primes up to N (enough for residue checks up to len <= N)
def _primes(limit):
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = bytearray(len(sieve[p * p::p]))
    return [i for i in range(limit + 1) if sieve[i]]

PRIMES = _primes(N)
# Primes up to N (for residue checks when len(tuple) can be up to N)
PRIMES_FOR_CHECK = [p for p in PRIMES if p <= N]

def _check_admissible(points, primes_to_check):
    """Return True if points is admissible w.r.t. primes_to_check."""
    pts = np.array(points, dtype=np.int64)
    for p in primes_to_check:
        if p > len(pts):
            break
        residues = np.mod(pts, p)
        if len(np.unique(residues)) == p:
            return False
    return True

def _greedy_delete(points, primes_to_check):
    """
    Greedy deletion: while not admissible, for each prime p <= len(points)
    that has all p residues covered, delete the least-populated residue class.
    Uses precomputed residue indices for speed.
    """
    points = list(points)
    # We need primes up to len(points)
    max_p = len(points)
    relevant_primes = [p for p in primes_to_check if p <= max_p]
    
    changed = True
    while changed:
        changed = False
        for p in relevant_primes:
            if p > len(points):
                continue
            # Count residues
            classes = {}
            for v in points:
                r = v % p
                if r not in classes:
                    classes[r] = []
                classes[r].append(v)
            
            if len(classes) == p:
                # All p residues covered; remove the smallest class
                smallest_class = min(classes.values(), key=len)
                smallest_set = set(smallest_class)
                points = [v for v in points if v not in smallest_set]
                changed = True
                break
    
    return points

def _fast_greedy(points, primes_to_check):
    """
    Faster greedy using numpy for residue counting.
    """
    points = np.array(points, dtype=np.int64)
    relevant_primes = [p for p in primes_to_check if p <= len(points)]
    
    changed = True
    while changed:
        changed = False
        for p in relevant_primes:
            if p > len(points):
                continue
            residues = np.mod(points, p)
            counts = np.bincount(residues, minlength=p)
            if np.all(counts > 0):
                # All p residues present; find min count residue
                min_r = int(np.argmin(counts))
                mask = residues != min_r
                points = points[mask]
                changed = True
                break
    
    return points.tolist()

def _random_start(seed_val):
    """Generate a random starting set."""
    rng = random.Random(seed_val)
    # Start with a random subset of [0, N]
    # Heuristic: start with about N/2 elements
    k = rng.randint(N // 3, N)
    points = rng.sample(range(N + 1), k)
    return points

def construct():
    start_time = time.time()
    best_points = None
    best_size = 0
    
    # Strategy: multiple random restarts with greedy deletion
    # Also try structured starts
    
    # First, try a structured start: all numbers, then greedy delete
    # This is expensive but gives a good baseline
    if time.time() - start_time < 10:
        # Start from all points
        points = list(range(N + 1))
        points = _fast_greedy(points, PRIMES_FOR_CHECK)
        if len(points) > best_size:
            best_size = len(points)
            best_points = points[:]
    
    # Now try random restarts
    attempt = 0
    while time.time() - start_time < DEADLINE - 5:
        attempt += 1
        # Vary the starting set
        if attempt <= 5:
            # Structured starts
            if attempt == 1:
                # Evens
                points = list(range(0, N + 1, 2))
            elif attempt == 2:
                # Odds
                points = list(range(1, N + 1, 2))
            elif attempt == 3:
                # Multiples of 3
                points = list(range(0, N + 1, 3))
            elif attempt == 4:
                # Numbers not divisible by 2 or 3
                points = [x for x in range(N + 1) if x % 2 != 0 and x % 3 != 0]
            else:
                # Numbers not divisible by 2, 3, or 5
                points = [x for x in range(N + 1) if x % 2 != 0 and x % 3 != 0 and x % 5 != 0]
        else:
            # Random starts with varying density
            seed_val = attempt * 12345 + 678
            density = 0.3 + 0.5 * ((attempt * 7919) % 100) / 100.0
            k = int(density * (N + 1))
            points = _random_start(seed_val)
            # Resample with specific density
            rng = random.Random(seed_val)
            k = max(100, int(density * (N + 1)))
            points = rng.sample(range(N + 1), min(k, N + 1))
        
        # Apply fast greedy
        points = _fast_greedy(points, PRIMES_FOR_CHECK)
        
        if len(points) > best_size:
            best_size = len(points)
            best_points = points[:]
            # Check if we should continue or if we're close to optimal
            # Theoretical max is around N * prod(1-1/p) for primes up to some bound
            # For fixed diameter, the max size is related to the covering problem
        
        # Check deadline
        if time.time() - start_time > DEADLINE - 10:
            break
    
    if best_points is None:
        # Fallback: return a small admissible set
        return [0, 1, 3]
    
    return best_points

# Pre-compute for speed
if __name__ == "__main__":
    result = construct()
    print(f"Result size: {len(result)}")
