"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _verify(nums):
    """Check all 2^n subset products are distinct."""
    n = len(nums)
    seen = set()
    for mask in range(1 << n):
        prod = 1
        for i in range(n):
            if mask & (1 << i):
                prod *= nums[i]
        if prod in seen:
            return False
        seen.add(prod)
    return True

def _prime_factorization(n):
    """Return dict of prime -> exponent for n."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def construct():
    deadline = time.time() + 80  # 80s safety margin
    
    # Primes in 2..40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Candidate composite numbers in 2..40 (non-prime)
    composites = [n for n in range(2, 41) if n not in primes]
    
    # For each composite, compute its prime factorization
    # We want to add composites that are prime powers (only one distinct prime factor)
    # and whose exponent doesn't create subset-sum collisions with existing exponents
    
    # Start with all primes
    result = list(primes)
    
    # Track exponents per prime for subset-sum distinctness
    # For each prime p, we track the set of exponents used
    prime_exponents = {}
    for p in primes:
        prime_exponents[p] = [1]  # Each prime contributes exponent 1
    
    # Try adding composites in order of "safety" (fewer prime factors first)
    # Sort composites by number of distinct prime factors, then by value
    def num_distinct_primes(n):
        return len(_prime_factorization(n))
    
    composites_sorted = sorted(composites, key=lambda x: (num_distinct_primes(x), x))
    
    # Greedily add composites that maintain distinct subset products
    # Use efficient verification via exponent vectors
    for comp in composites_sorted:
        if time.time() > deadline - 5:
            break
        
        # Check if comp is a prime power (single distinct prime factor)
        pf = _prime_factorization(comp)
        if len(pf) != 1:
            continue  # Skip composites with multiple prime factors
        
        p = list(pf.keys())[0]
        e = pf[p]
        
        # Check if adding exponent e to prime p's exponent set maintains
        # distinct subset sums
        existing = prime_exponents.get(p, [])
        new_exponents = existing + [e]
        
        # Check all subset sums are distinct
        n_exp = len(new_exponents)
        sums = set()
        for mask in range(1 << n_exp):
            s = 0
            for i in range(n_exp):
                if mask & (1 << i):
                    s += new_exponents[i]
            if s in sums:
                break
            sums.add(s)
        else:
            # All subset sums distinct - safe to add
            prime_exponents[p] = new_exponents
            result.append(comp)
    
    # Verify the full set
    if _verify(result):
        return result
    
    # Fallback: known valid set of 16
    fallback = primes + [4, 16, 9, 25]
    if _verify(fallback):
        return fallback
    
    return primes
# EVOLVE-BLOCK-END
