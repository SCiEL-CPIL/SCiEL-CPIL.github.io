"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import sys
from functools import lru_cache
from itertools import combinations

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _largest_prime_factor(n):
    """Return the largest prime factor of n."""
    if n < 2:
        return 1
    lp = 1
    d = 2
    while d * d <= n:
        while n % d == 0:
            lp = d
            n //= d
        d += 1
    if n > 1:
        lp = max(lp, n)
    return lp

def _check_valid(nums):
    """Check if all subset products are distinct. Returns True/False."""
    k = len(nums)
    products = set()
    products.add(1)
    for i in range(k):
        new_prods = []
        for p in products:
            new_p = p * nums[i]
            if new_p in products:
                return False
            new_prods.append(new_p)
        products.update(new_prods)
    return True

def _check_valid_fast(nums):
    """
    Fast check using prime factor exponent vectors.
    Two products are equal iff their exponent vectors are equal.
    """
    k = len(nums)
    # Build exponent vectors for each number
    primes = set()
    for n in nums:
        temp = n
        d = 2
        while d * d <= temp:
            if temp % d == 0:
                primes.add(d)
                while temp % d == 0:
                    temp //= d
            d += 1
        if temp > 1:
            primes.add(temp)
    
    prime_list = sorted(primes)
    prime_idx = {p: i for i, p in enumerate(prime_list)}
    n_primes = len(prime_list)
    
    # Build exponent vector for each number
    exps = []
    for n in nums:
        vec = [0] * n_primes
        temp = n
        for i, p in enumerate(prime_list):
            while temp % p == 0:
                vec[i] += 1
                temp //= p
        exps.append(tuple(vec))
    
    # Now check subset sum distinctness
    # We need all 2^k subset sums to be distinct
    # This is equivalent to: no two different subsets have the same sum
    # Use a set of tuples for efficiency
    products = set()
    products.add((0,) * n_primes)
    for i in range(k):
        new_prods = []
        for p in products:
            new_p = tuple(p[j] + exps[i][j] for j in range(n_primes))
            if new_p in products:
                return False
            new_prods.append(new_p)
        products.update(new_prods)
    return True

def _try_greedy(candidates, order):
    """Greedy: add candidates in given order if valid."""
    result = []
    for c in order:
        if c not in result:
            test = result + [c]
            if _check_valid(test):
                result.append(c)
    return result

def construct():
    """Return a list of up to 18 distinct integers in 2..40 with distinct subset products."""
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin
    
    # All candidate numbers
    all_nums = list(range(2, 41))
    
    # Primes in range
    primes = [n for n in all_nums if _is_prime(n)]
    
    # Prime powers in range
    prime_powers = []
    for n in all_nums:
        if n not in primes:
            # Check if n is a prime power
            is_pp = False
            temp = n
            d = 2
            while d * d <= temp:
                if temp % d == 0:
                    orig_d = d
                    while temp % d == 0:
                        temp //= d
                    if temp == 1:
                        is_pp = True
                        break
                d += 1
            if not is_pp and temp > 1:
                # temp is a prime, check if n is a power of it
                if n == temp:
                    pass  # it's prime, already handled
            if is_pp:
                prime_powers.append(n)
    
    # Build a few different orderings for greedy
    orderings = []
    
    # Ordering 1: primes first, then prime powers, then rest
    order1 = primes + [x for x in prime_powers if x not in primes] + [x for x in all_nums if x not in primes and x not in prime_powers]
    orderings.append(order1)
    
    # Ordering 2: by largest prime factor (ascending)
    order2 = sorted(all_nums, key=lambda x: _largest_prime_factor(x))
    orderings.append(order2)
    
    # Ordering 3: by number of prime factors (ascending)
    def _num_prime_factors(n):
        count = 0
        temp = n
        d = 2
        while d * d <= temp:
            while temp % d == 0:
                count += 1
                temp //= d
            d += 1
        if temp > 1:
            count += 1
        return count
    order3 = sorted(all_nums, key=lambda x: _num_prime_factors(x))
    orderings.append(order3)
    
    # Ordering 4: just ascending
    orderings.append(all_nums)
    
    # Ordering 5: descending
    orderings.append(sorted(all_nums, reverse=True))
    
    # Run greedy with each ordering
    best = []
    for order in orderings:
        if time.time() > deadline:
            break
        result = _try_greedy(all_nums, order)
        if len(result) > len(best):
            best = result
    
    # Now try to improve by adding more elements
    # Try random restarts with randomized orderings
    rng = random.Random(42)
    max_iter = 200
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Random permutation
        random_order = all_nums[:]
        rng.shuffle(random_order)
        
        result = _try_greedy(all_nums, random_order)
        if len(result) > len(best):
            best = result
            if len(best) >= 18:
                break
    
    # Try local search: try to add elements not in best
    if len(best) < 18:
        for _ in range(500):
            if time.time() > deadline:
                break
            added = False
            for c in all_nums:
                if c not in best:
                    test = best + [c]
                    if _check_valid(test):
                        best.append(c)
                        added = True
                        if len(best) >= 18:
                            break
            if not added:
                break
    
    # Final verification
    if not _check_valid(best):
        # Fallback to primes
        best = primes
    
    # Ensure types and ranges
    result = []
    for x in best:
        if isinstance(x, int) and not isinstance(x, bool) and 2 <= x <= 40:
            result.append(x)
    
    # Ensure distinct
    seen = set()
    unique = []
    for x in result:
        if x not in seen:
            seen.add(x)
            unique.append(x)
    
    return unique
# EVOLVE-BLOCK-END
