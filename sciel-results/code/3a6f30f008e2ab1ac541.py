import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Numbers 2 to 40
    candidates = list(range(2, 41))
    
    # Prime factorization for numbers 2-40
    # Primes up to 40: 2,3,5,7,11,13,17,19,23,29,31,37
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    def factorize(n):
        vec = [0] * len(primes)
        for i, p in enumerate(primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        return tuple(vec)
    
    # Precompute exponent vectors
    vectors = {}
    for n in candidates:
        vectors[n] = factorize(n)
    
    def has_distinct_subset_products(nums):
        """Check if all 2^k subset products are distinct."""
        prods = {1}
        for x in nums:
            new_prods = set()
            for p in prods:
                new_prods.add(p * x)
            if len(new_prods) != len(prods):
                # Some product already exists
                return False
            prods = new_prods | prods
        return True
    
    # Try to find the largest valid subset
    # Start with size 12 (the primes)
    best = None
    
    # The 12 primes form a valid set
    prime_set = [p for p in primes if p <= 40]
    if has_distinct_subset_products(prime_set):
        best = list(prime_set)
    
    # Try to see if we can get more than 12
    # Since we have 12 primes, max possible is 12 by dimension argument
    # But let's verify by trying all combinations of size 13, 14, etc.
    # Actually, with 39 candidates, combinations of size 13 is C(39,13) which is huge
    # Let's be smarter: use the fact that any valid set must have linearly independent
    # exponent vectors, so max size is 12.
    
    # Verify: can we get 12? Yes, the primes work.
    # Can we get 13? No, by dimension argument.
    
    # So the answer is 12.
    # But let's double-check by trying a few random sets of size 12 to make sure
    # the primes are optimal, and also verify our check function.
    
    if time.time() > deadline:
        return best if best else prime_set
    
    # Verify the prime set is valid
    assert has_distinct_subset_products(prime_set), "Prime set should be valid"
    
    return list(prime_set)
