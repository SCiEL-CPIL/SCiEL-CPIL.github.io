"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

sys.setrecursionlimit(10000)

_START_TIME = time.time()
_DEADLINE = _START_TIME + 55.0  # Safety margin for 60s limit

# Precompute primes up to 40
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

# All primes in 2..40
_ALL_PRIMES = [p for p in range(2, 41) if _is_prime(p)]
# Primes that appear in factorizations of numbers 2..40
_RELEVANT_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def _factorize(n):
    """Return dict of prime -> exponent for n."""
    factors = {}
    for p in _RELEVANT_PRIMES:
        while n % p == 0:
            factors[p] = factors.get(p, 0) + 1
            n //= p
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

# Precompute factorizations for 2..40
_FACTORIZATIONS = {}
for i in range(2, 41):
    _FACTORIZATIONS[i] = _factorize(i)

# Precompute prime exponent vectors as tuples for fast comparison
_PRIME_TO_IDX = {p: i for i, p in enumerate(_RELEVANT_PRIMES)}
_EXP_VECTORS = {}
for i in range(2, 41):
    vec = [0] * len(_RELEVANT_PRIMES)
    for p, e in _FACTORIZATIONS[i].items():
        if p in _PRIME_TO_IDX:
            vec[_PRIME_TO_IDX[p]] = e
    _EXP_VECTORS[i] = tuple(vec)

def _can_add_fast(products, x):
    """
    Check if adding x to a set with existing subset products 'products'
    keeps all subset products distinct.
    """
    for p in products:
        if p * x in products:
            return False
    return True

def _add_number(products, x):
    """Return new set of products after adding x."""
    new_prods = set(products)
    for p in products:
        new_prods.add(p * x)
    return new_prods

def _verify(nums):
    """Verify all subset products are distinct."""
    prods = {1}
    for x in nums:
        new_prods = set()
        for p in prods:
            new_prods.add(p * x)
        if len(prods | new_prods) != len(prods) + len(new_prods):
            return False
        prods = prods | new_prods
    return True

def construct():
    """
    Find the largest set of distinct integers in 2..40 such that all
    2^k subset products (including empty product = 1) are distinct.
    """
    candidates = list(range(2, 41))
    primes = [x for x in candidates if _is_prime(x)]
    composites = [x for x in candidates if not _is_prime(x)]
    
    # Key insight: prime powers (p^e where e>=2) have exponent vectors
    # with a single non-zero component >= 2, which cannot be formed by
    # sum/difference of prime unit vectors (which have components in {-1,0,1}).
    # So prime powers are "safe" to add to the prime set.
    # However, interactions between multiple prime powers must also be checked.
    
    # Strategy: Start with all primes, then greedily add composites.
    # Order composites by "safety": prime powers first (most likely to work),
    # then others by number of distinct prime factors.
    
    def num_distinct_prime_factors(x):
        return len(_FACTORIZATIONS[x])
    
    def is_prime_power(x):
        f = _FACTORIZATIONS[x]
        return len(f) == 1 and list(f.values())[0] >= 2
    
    # Sort composites: prime powers first, then by distinct prime factor count
    def composite_priority(x):
        if is_prime_power(x):
            return (0, 0, x)
        return (1, num_distinct_prime_factors(x), x)
    
    sorted_composites = sorted(composites, key=composite_priority)
    
    # Greedy: primes first, then composites in priority order
    products = {1}
    selected = list(primes)
    for x in primes:
        products = _add_number(products, x)
    
    for x in sorted_composites:
        if _can_add_fast(products, x):
            products = _add_number(products, x)
            selected.append(x)
    
    best_set = list(selected)
    best_size = len(best_set)
    
    # Now try to improve by trying different orderings and local search
    # Try multiple greedy orderings
    for trial in range(20):
        if time.time() > _DEADLINE - 5:
            break
        # Shuffle composites with some randomness in priority
        import random
        random.seed(trial)
        shuffled_comps = list(composites)
        # Partially shuffle: keep prime powers somewhat prioritized
        pp = [x for x in shuffled_comps if is_prime_power(x)]
        non_pp = [x for x in shuffled_comps if not is_prime_power(x)]
        random.shuffle(pp)
        random.shuffle(non_pp)
        order = pp + non_pp
        
        prods = {1}
        sel = list(primes)
        for x in primes:
            prods = _add_number(prods, x)
        for x in order:
            if _can_add_fast(prods, x):
                prods = _add_number(prods, x)
                sel.append(x)
        
        if len(sel) > best_size:
            best_size = len(sel)
            best_set = list(sel)
    
    # Local search: try to add more by removing and re-adding
    def _local_improve(nums, deadline):
        nums = list(nums)
        products = {1}
        for x in nums:
            products = _add_number(products, x)
        
        candidates = [x for x in range(2, 41) if x not in nums]
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            # Try adding a candidate
            for x in candidates:
                if time.time() > deadline:
                    break
                if _can_add_fast(products, x):
                    products = _add_number(products, x)
                    nums.append(x)
                    candidates.remove(x)
                    improved = True
                    break
            
            # Try 1-for-1 swap
            if not improved and len(nums) >= 1:
                for i in range(len(nums)):
                    if time.time() > deadline:
                        break
                    remaining = nums[:i] + nums[i+1:]
                    rem_prods = {1}
                    for x in remaining:
                        rem_prods = _add_number(rem_prods, x)
                    
                    # Try to add one that fits
                    for x in candidates:
                        if time.time() > deadline:
                            break
                        if _can_add_fast(rem_prods, x):
                            # Check if we can then add another
                            new_prods = _add_number(rem_prods, x)
                            added_second = False
                            for y in candidates:
                                if y == x:
                                    continue
                                if _can_add_fast(new_prods, y):
                                    added_second = True
                                    break
                            if added_second:
                                nums = remaining + [x]
                                products = new_prods
                                if x in candidates:
                                    candidates.remove(x)
                                improved = True
                                break
                    if improved:
                        break
        
        return nums
    
    if time.time() < _DEADLINE - 3:
        improved = _local_improve(best_set, _DEADLINE - 2)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
    
    # Verify
    if not _verify(best_set):
        best_set = primes
    
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
