"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def construct():
    deadline = time.time() + 280  # safety margin before 330s kill
    points = list(range(N + 1))
    
    # Precompute primes up to N+1
    primes = _primes(N + 1)
    
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        k = len(points)
        # Process primes in DESCENDING order: large primes have nearly-singleton
        # residue classes, so removing the smallest class is very cheap
        for p in reversed(primes):
            if p > k:
                continue
            # Build residue class counts
            classes = {}
            for v in points:
                r = v % p
                classes[r] = classes.get(r, 0) + 1
            if len(classes) == p:
                # All p residue classes are present; remove the smallest class
                smallest_r = min(classes, key=lambda r: classes[r])
                remove_set = set(v for v in points if v % p == smallest_r)
                points = [v for v in points if v not in remove_set]
                changed = True
                break
    return points
# EVOLVE-BLOCK-END
