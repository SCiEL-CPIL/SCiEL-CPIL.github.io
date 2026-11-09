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
    S = list(range(N + 1))  # 0 to 5000
    changed = True
    while changed:
        changed = False
        k = len(S)
        primes_to_check = _primes(k)
        for p in primes_to_check:
            # Count elements in each residue class mod p
            classes = [0] * p
            for v in S:
                classes[v % p] += 1
            # If all classes are non-empty, we need to remove one
            if min(classes) > 0:
                # Remove the least-populated class
                min_count = min(classes)
                # Find which residue has min count
                target_residue = classes.index(min_count)
                S = [v for v in S if v % p != target_residue]
                changed = True
                break
    return S
