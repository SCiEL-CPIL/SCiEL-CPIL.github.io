import time

N = 5000
_DEADLINE = time.time() + 320.0

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, ok in enumerate(sieve) if ok]

def construct():
    # Start with all integers in [0, N]
    points = list(range(N + 1))
    primes = _primes(N)
    
    while True:
        if time.time() > _DEADLINE:
            return points
            
        k = len(points)
        if k == 0:
            return points
            
        # Find the smallest prime p <= k such that points covers all residues mod p
        # If no such prime exists, points is admissible.
        removed = False
        for p in primes:
            if p > k:
                break
                
            # Count residues
            counts = [0] * p
            for v in points:
                counts[v % p] += 1
                
            min_count = min(counts)
            if min_count > 0:
                # All residues are present. We must remove one residue class.
                # Remove the one with the fewest elements to maximize remaining size.
                target_residue = counts.index(min_count)
                points = [v for v in points if v % p != target_residue]
                removed = True
                break
                
        if not removed:
            return points
