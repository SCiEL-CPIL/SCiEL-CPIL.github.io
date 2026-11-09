"""Rewritten program: search for a large set of distinct integers in 2..40 with distinct subset products."""
# EVOLVE-BLOCK-START
import time
import itertools

DEADLINE = 90.0
MAX_TIME = 60.0  # safety margin before 90s kill

def _factorize(n, primes):
    """Return dict of prime->exponent for n."""
    f = {}
    for p in primes:
        if p * p > n:
            break
        while n % p == 0:
            f[p] = f.get(p, 0) + 1
            n //= p
    if n > 1:
        f[n] = f.get(n, 0) + 1
    return f

def _subset_products_distinct(nums, primes):
    """Check if all 2^k subset products of nums are distinct.
    Uses prime exponent vectors: two subsets have same product iff they have same exponent vector.
    So we need all 2^k exponent vectors to be distinct.
    """
    k = len(nums)
    if k == 0:
        return True
    # Represent each number as a vector of exponents over primes
    # Two subsets have the same product iff the sum of their exponent vectors is the same.
    # This is equivalent to: the 2^k sums of subsets of the exponent vectors are all distinct.
    # This happens iff the exponent vectors are linearly independent over the field of rationals? 
    # No, not quite. We need that no non-empty subset has sum zero... but sums are non-negative.
    # Actually, two different subsets A and B have the same product iff the sum of vectors in A equals sum in B.
    # This means the difference (A \ B) and (B \ A) have the same sum of vectors.
    # So we need: no two disjoint non-empty subsets have the same sum of exponent vectors.
    # Equivalently: all 2^k subset sums of the exponent vectors are distinct.
    
    # For efficiency with small k (up to ~18), we can enumerate all subset sums.
    # But 2^18 = 262144, and each sum is a dict or tuple of exponents.
    # Let's use a set of tuples.
    
    # First, get the prime list that appears in any of the numbers
    all_primes = set()
    vecs = []
    for num in nums:
        f = _factorize(num, primes)
        all_primes.update(f.keys())
        vecs.append(f)
    
    prime_list = sorted(all_primes)
    prime_index = {p: i for i, p in enumerate(prime_list)}
    dim = len(prime_list)
    
    # Convert each factor dict to a tuple of exponents
    exp_vecs = []
    for f in vecs:
        v = [0] * dim
        for p, e in f.items():
            v[prime_index[p]] = e
        exp_vecs.append(tuple(v))
    
    # Enumerate all subset sums
    # Use iterative approach: start with {0}, for each vector, add it to all existing sums
    sums = {tuple([0]*dim)}
    for v in exp_vecs:
        new_sums = set()
        for s in sums:
            new_sums.add(tuple(s[i] + v[i] for i in range(dim)))
        sums = new_sums
        if len(sums) < (1 << len(exp_vecs[:exp_vecs.index(v)+1])):
            # Duplicate found
            return False
    
    return len(sums) == (1 << k)

def construct():
    start_time = time.time()
    
    # Generate primes up to 40
    primes = []
    for n in range(2, 41):
        is_prime = True
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(n)
    
    # Candidate numbers: all integers from 2 to 40
    candidates = list(range(2, 41))
    
    # Strategy: greedy search with backtracking
    # We want to maximize the size of the set.
    # Start with a greedy selection, then try to improve.
    
    # Greedy: add numbers in some order, check if subset products remain distinct.
    # Try multiple orderings.
    
    best_set = []
    
    # Ordering 1: primes first (they are "independent" in some sense)
    ordering1 = primes + [n for n in candidates if n not in primes]
    
    # Ordering 2: numbers with fewer prime factors first
    def num_prime_factors(n):
        f = _factorize(n, primes)
        return sum(f.values())
    
    ordering2 = sorted(candidates, key=lambda x: (num_prime_factors(x), x))
    
    # Ordering 3: just natural order
    ordering3 = candidates
    
    # Ordering 4: reverse natural
    ordering4 = list(reversed(candidates))
    
    for ordering in [ordering1, ordering2, ordering3, ordering4]:
        if time.time() - start_time > MAX_TIME:
            break
        current = []
        for num in ordering:
            if time.time() - start_time > MAX_TIME:
                break
            test = current + [num]
            if _subset_products_distinct(test, primes):
                current = test
        if len(current) > len(best_set):
            best_set = current
    
    # Now try to improve: remove one element and try to add others
    # This is a local search / hill climbing
    improved = True
    while improved and (time.time() - start_time < MAX_TIME):
        improved = False
        best_size = len(best_set)
        for i in range(len(best_set)):
            if time.time() - start_time > MAX_TIME:
                break
            # Remove best_set[i]
            reduced = best_set[:i] + best_set[i+1:]
            # Try to add each candidate not in reduced
            for num in candidates:
                if num in reduced:
                    continue
                if time.time() - start_time > MAX_TIME:
                    break
                test = reduced + [num]
                if len(test) > best_size:
                    if _subset_products_distinct(test, primes):
                        best_set = test
                        best_size = len(test)
                        improved = True
                        break
                elif len(test) == best_size:
                    # Same size, but maybe leads to better future? Skip for simplicity
                    pass
    
    # Ensure we return a list of Python ints
    result = [int(x) for x in best_set]
    return result
# EVOLVE-BLOCK-END
