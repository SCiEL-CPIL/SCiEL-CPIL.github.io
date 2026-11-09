N = 5000

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
    S = list(range(N + 1))
    primes = _primes(N)
    # Sort in descending order
    primes.sort(reverse=True)
    
    for p in primes:
        k = len(S)
        if p > k:
            # Since primes are sorted descending, and k only decreases,
            # if p > k, all subsequent primes will be < p.
            # But we still need to check if they are <= k.
            # So we cannot break. We just skip this p.
            continue
        
        # Count elements in each residue class mod p
        classes = [0] * p
        for v in S:
            classes[v % p] += 1
        
        # If all classes are non-empty, we need to remove one
        if min(classes) > 0:
            min_count = min(classes)
            target_residue = classes.index(min_count)
            S = [v for v in S if v % p != target_residue]
            
    return S
