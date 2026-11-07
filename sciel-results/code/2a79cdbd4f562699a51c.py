import time
import sys

sys.setrecursionlimit(10000)

_START_TIME = time.time()
_DEADLINE = _START_TIME + 55.0

def _get_prime_exponents(n, primes):
    """Return list of exponents of primes in n."""
    exponents = [0] * len(primes)
    for i, p in enumerate(primes):
        while n % p == 0:
            exponents[i] += 1
            n //= p
    return exponents

def _can_add(products, x, primes):
    """
    Check if adding x to a set with existing subset products 'products'
    keeps all subset products distinct.
    """
    new_prods = []
    for p in products:
        np = p * x
        if np in products:
            return False, None
        new_prods.append(np)
    # Check for collisions within new_prods?
    # Since x is fixed, p1*x == p2*x implies p1==p2.
    # So new_prods are distinct if products are distinct.
    # We just need to check intersection with existing products.
    # The loop above does that.
    return True, new_prods

def _add_number(products, x):
    """Return new list of products after adding x."""
    new_prods = list(products)
    for p in products:
        new_prods.append(p * x)
    return new_prods

def _is_valid_set(nums, primes):
    """Verify that all subset products of nums are distinct."""
    if not nums:
        return True
    products = {1}
    for x in nums:
        new_prods = set()
        for p in products:
            new_prods.add(p * x)
        # Check intersection
        if products & new_prods:
            return False
        products = products | new_prods
    return True

def construct():
    """
    Find the largest set of distinct integers in 2..40 such that all
    2^k subset products (including empty product = 1) are distinct.
    """
    candidates = list(range(2, 41))
    
    # Identify primes
    primes = []
    for x in candidates:
        is_p = True
        for i in range(2, int(x**0.5) + 1):
            if x % i == 0:
                is_p = False
                break
        if is_p:
            primes.append(x)
            
    # Define heuristic ordering
    # 1. Primes
    # 2. Squares of primes
    # 3. Cubes of primes
    # 4. Fourth powers
    # 5. Fifth powers
    # 6. Other composites sorted by number of prime factors (ascending)
    
    def get_order_key(x):
        if x in primes:
            return (0, x)
        
        # Check if power of a prime
        for p in primes:
            if x == p**2:
                return (1, x)
            if x == p**3:
                return (2, x)
            if x == p**4:
                return (3, x)
            if x == p**5:
                return (4, x)
        
        # Count prime factors with multiplicity
        count = 0
        n = x
        for p in primes:
            while n % p == 0:
                count += 1
                n //= p
        # If n > 1, it has a prime factor > sqrt(40) or is prime itself (but primes handled)
        # Actually if n>1 and not divisible by small primes, it might be a large prime?
        # But x <= 40. Primes are 2..37.
        # If n > 1, it's a prime factor > 37? No, 40 < 37*2.
        # So n must be 1 or a prime > 37? No.
        # Just count remaining as 1 if n>1.
        if n > 1:
            count += 1
            
        return (5, count, x)

    ordered_candidates = sorted(candidates, key=get_order_key)
    
    # Greedy construction
    products = {1}
    selected = []
    
    for x in ordered_candidates:
        if time.time() > _DEADLINE:
            break
            
        # Check if we can add x
        can, new_prods_list = _can_add(products, x, primes)
        if can:
            # Update products set
            # We need to add all new products to the set
            # _can_add returned a list of new products, but we need the union
            # Actually, _can_add only checked collisions.
            # We need to construct the new set.
            new_products_set = set(products)
            for p in products:
                new_products_set.add(p * x)
            
            # Verify no collisions in new_products_set?
            # _can_add ensured p*x not in products.
            # And p1*x != p2*x.
            # So new_products_set is valid.
            
            products = new_products_set
            selected.append(x)
            
            # If we have a very large set, we might run out of time verifying
            # but the set size is at most 39, so 2^39 is too big to store.
            # However, we are only adding one by one.
            # The set of products grows by factor 2 each time.
            # If we have 20 numbers, we have 2^20 ~ 10^6 products.
            # Storing 10^6 big ints might be slow/memory heavy.
            # But 40 is small.
            
    # Verify the result
    if not _is_valid_set(selected, primes):
        # Fallback to just primes
        selected = primes
        
    return [int(x) for x in selected]
