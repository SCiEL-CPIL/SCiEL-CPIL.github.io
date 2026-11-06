"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def construct():
    # Start with all integers in [0, N]
    points = set(range(N + 1))
    
    # Process primes in increasing order
    # For each prime p <= len(points), ensure not all residue classes mod p are covered
    # If all are covered, remove all elements from the least-populated residue class
    while True:
        k = len(points)
        if k == 0:
            break
        primes = _primes(k)
        changed = False
        for p in primes:
            # Count elements in each residue class mod p
            classes = {}
            for v in points:
                r = v % p
                classes[r] = classes.get(r, 0) + 1
            if len(classes) == p:
                # All p residue classes are covered; remove the least-populated class
                min_count = min(classes.values())
                min_class = min(classes.keys(), key=lambda r: classes[r])
                # Remove all elements in that residue class
                points = {v for v in points if v % p != min_class}
                changed = True
                break  # Restart since points changed
        if not changed:
            break
    
    return sorted(points)
# EVOLVE-BLOCK-END
