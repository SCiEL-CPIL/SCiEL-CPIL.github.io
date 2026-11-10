"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 40, 'maximum_size': 18}

def _get_primes(n):
    """Return all primes <= n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n+1, i):
                sieve[j] = False
    return [i for i in range(2, n+1) if sieve[i]]

def _prime_factorization(num, primes):
    """Return tuple of exponents for each prime in primes."""
    exponents = []
    for p in primes:
        count = 0
        while num % p == 0:
            count += 1
            num //= p
        exponents.append(count)
    return tuple(exponents)

def _build_vectors(candidates, primes):
    """Pre-compute prime exponent vectors for all candidates."""
    vectors = {}
    for cand in candidates:
        vectors[cand] = _prime_factorization(cand, primes)
    return vectors

def construct():
    """
    Find the largest subset of {2, ..., 40} such that all subset products are distinct.
    Uses multiple random restarts with greedy selection and prime exponent vectors.
    """
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin within 60s limit
    
    candidates = list(range(2, P['n'] + 1))
    primes = _get_primes(P['n'])
    vectors = _build_vectors(candidates, primes)
    num_primes = len(primes)
    
    best_set = []
    best_len = 0
    
    # Pre-compute a simple complexity metric for sorting heuristics
    def complexity(num):
        return sum(vectors[num])

    complexities = {x: complexity(x) for x in candidates}

    num_restarts = 100000  # High number, but loop breaks on time
    
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Choose a strategy for ordering
        r = random.random()
        if r < 0.2:
            ordered = sorted(candidates, key=lambda x: (complexities[x], x))
        elif r < 0.4:
            ordered = sorted(candidates)
        elif r < 0.6:
            ordered = sorted(candidates, reverse=True)
        else:
            ordered = candidates[:]
            random.shuffle(ordered)
            
        current_set = []
        # products is a set of tuples (prime exponent vectors)
        zero_vec = tuple([0] * num_primes)
        products = {zero_vec}
        
        for cand in ordered:
            cand_vec = vectors[cand]
            
            # Check if adding cand keeps all products distinct
            # New products are {p + cand_vec for p in products}
            # Collision if any p + cand_vec is already in products
            
            valid = True
            for p in products:
                new_p = tuple(p[i] + cand_vec[i] for i in range(num_primes))
                if new_p in products:
                    valid = False
                    break
            
            if valid:
                current_set.append(cand)
                # Update products
                new_prods = [tuple(p[i] + cand_vec[i] for i in range(num_primes)) for p in products]
                products.update(new_prods)
                
                if len(current_set) > best_len:
                    best_len = len(current_set)
                    best_set = current_set[:]

    # Fallback if no set found (should not happen)
    if not best_set:
        best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    return list(best_set)
# EVOLVE-BLOCK-END
