import itertools

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def _distinct_subset_sums(exponents):
    """Check if all subset sums of exponents are distinct."""
    sums = set()
    for r in range(len(exponents) + 1):
        for combo in itertools.combinations(exponents, r):
            s = sum(combo)
            if s in sums:
                return False
            sums.add(s)
    return True

def _best_exponent_subset(max_exp):
    """Find largest subset of {1..max_exp} with distinct subset sums."""
    candidates = list(range(1, max_exp + 1))
    # Try sizes from max_exp down to 1
    for size in range(len(candidates), 0, -1):
        for subset in itertools.combinations(candidates, size):
            if _distinct_subset_sums(subset):
                return subset
    return []

def construct():
    max_val = 40
    primes = _primes_up_to(max_val)
    selected = []
    
    for p in primes:
        # Find max exponent M such that p^M <= max_val
        m = 0
        val = 1
        while val * p <= max_val:
            val *= p
            m += 1
        
        if m == 0:
            continue
            
        best_exps = _best_exponent_subset(m)
        
        for e in best_exps:
            selected.append(p ** e)
            
    return selected
