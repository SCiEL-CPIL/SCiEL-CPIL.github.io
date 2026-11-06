"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _primes_upto(n):
    """Generate all primes <= n using sieve of Eratosthenes."""
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n+1) if sieve[i]]

PRIMES = _primes_upto(1000)
NUM_PRIMES = len(PRIMES)  # 168

def _score(residues, limit=100000):
    """Compute the largest L such that every integer 1..L is covered by >= 2 congruences."""
    # For efficiency, we count coverage for each x in 1..limit
    # But doing this naively is O(limit * num_primes) which is 100000 * 168 = 16.8M, acceptable
    # Actually, let's use a difference array or just direct counting for small ranges.
    
    # Since limit can be up to 100000, and we have 168 primes, 
    # we can compute coverage count for each x in 1..limit by checking each prime.
    # But that's 16.8M operations which is fine in Python if optimized.
    
    # Alternative: for each prime, mark its residue class in 1..limit
    # Use an array of length limit+1, increment for each x in residue class
    
    count = [0] * (limit + 1)
    for p, r in zip(PRIMES, residues):
        # x = r (mod p), so x = r, r+p, r+2p, ... <= limit
        # But r could be 0, meaning x = 0, p, 2p, ... 
        # We care about x in 1..limit
        start = r if r > 0 else p  # first positive multiple
        if start > limit:
            continue
        for x in range(start, limit + 1, p):
            count[x] += 1
    
    # Find largest L such that count[x] >= 2 for all x in 1..L
    L = 0
    for x in range(1, limit + 1):
        if count[x] >= 2:
            L = x
        else:
            break
    return L

def _fast_score(residues, limit=20000):
    """Faster scoring for smaller limits, used during search."""
    count = [0] * (limit + 1)
    for p, r in zip(PRIMES, residues):
        start = r if r > 0 else p
        if start > limit:
            continue
        for x in range(start, limit + 1, p):
            count[x] += 1
    
    L = 0
    for x in range(1, limit + 1):
        if count[x] >= 2:
            L = x
        else:
            break
    return L

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Initialize with the seed solution
    best_residues = [((i + 2) // 2) % p for i, p in enumerate(PRIMES)]
    best_score = _score(best_residues, limit=100000)
    
    # Local search: try random perturbations
    random.seed(42)
    iterations = 0
    
    while time.time() < deadline:
        iterations += 1
        
        # Create a perturbation of the current best
        candidate = best_residues[:]
        
        # Randomly modify 1-3 residues
        num_changes = random.randint(1, 3)
        indices = random.sample(range(NUM_PRIMES), num_changes)
        for idx in indices:
            p = PRIMES[idx]
            candidate[idx] = random.randint(0, p - 1)
        
        # Quick score with smaller limit
        quick_score = _fast_score(candidate, limit=min(5000, best_score + 2000))
        
        if quick_score >= best_score:
            # Full score
            full_score = _score(candidate, limit=100000)
            if full_score > best_score:
                best_score = full_score
                best_residues = candidate
        
        # Also try hill climbing: single-residue changes
        if iterations % 10 == 0:
            for idx in random.sample(range(NUM_PRIMES), min(10, NUM_PRIMES)):
                p = PRIMES[idx]
                original = best_residues[idx]
                # Try a few nearby values
                for delta in [-2, -1, 1, 2]:
                    new_val = (original + delta) % p
                    candidate2 = best_residues[:]
                    candidate2[idx] = new_val
                    quick_score2 = _fast_score(candidate2, limit=min(5000, best_score + 2000))
                    if quick_score2 >= best_score:
                        full_score2 = _score(candidate2, limit=100000)
                        if full_score2 > best_score:
                            best_score = full_score2
                            best_residues = candidate2
                            break
    
    return best_residues

# EVOLVE-BLOCK-END
