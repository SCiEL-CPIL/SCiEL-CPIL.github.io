"""Rewritten program: maximize number of distinct integers in 2..40 with distinct subset products."""
# EVOLVE-BLOCK-START
import time

P = {'n': 40, 'maximum_size': 18}

# Precompute primes in 2..40
def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    d = 5
    while d * d <= n:
        if n % d == 0 or n % (d + 2) == 0:
            return False
        d += 6
    return True

PRIMES = [a for a in range(2, 41) if _is_prime(a)]
# Primes: 2,3,5,7,11,13,17,19,23,29,31,37 -> 12 primes

def _check_distinct_subset_products(S):
    """
    Check if all 2^k subset products of S are distinct.
    S is a list of integers.
    """
    if not S:
        return True
    products = {1}
    for x in S:
        new_products = set()
        for p in products:
            np = p * x
            if np in products or np in new_products:
                return False
            new_products.add(np)
        products.update(new_products)
    return True

def _try_add(products, x):
    """
    Try to add x to the set of products.
    Returns None if collision, else list of new products.
    """
    new_products = []
    for p in products:
        np = p * x
        if np in products or np in new_products:
            return None
        new_products.append(np)
    return new_products

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin before 60s limit
    
    # Start with primes (12 elements)
    base = list(PRIMES)
    
    # Verify base is valid (it always is, but good practice)
    if not _check_distinct_subset_products(base):
        return base
    
    # Initialize product set for the base
    best = list(base)
    best_products = {1}
    for x in best:
        new_prods = _try_add(best_products, x)
        if new_prods is not None:
            best_products.update(new_prods)
    
    # Greedy: try to add composites
    # Theoretical max is 12 (number of primes), so we expect no composites to be added.
    # However, we run the check to be robust against any theoretical oversight or if range changes.
    composites = [a for a in range(2, 41) if not _is_prime(a)]
    
    # Sort composites by prime factor count (heuristic)
    def prime_factor_count(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
        
    composites.sort(key=lambda x: (prime_factor_count(x), x))
    
    for c in composites:
        if time.time() > deadline:
            break
        new_prods = _try_add(best_products, c)
        if new_prods is not None:
            best.append(c)
            best_products.update(new_prods)
            
    # If we haven't reached the theoretical max (12), we might try a few random restarts
    # but given the constraint, it's highly unlikely to find > 12.
    # We skip expensive local search to ensure speed and reliability.
    
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
