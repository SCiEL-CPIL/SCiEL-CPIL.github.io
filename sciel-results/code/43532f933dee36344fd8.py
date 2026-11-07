"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

sys.setrecursionlimit(10000)

_START_TIME = time.time()
_DEADLINE = _START_TIME + 55.0  # Safety margin for 60s limit

# Precompute prime exponent vectors for numbers 2..40
_PRIMES_UP_TO_40 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
_NUM_PRIMES = len(_PRIMES_UP_TO_40)

def _prime_exp_vector(n):
    """Return tuple of exponents of primes in factorization of n."""
    vec = [0] * _NUM_PRIMES
    for i, p in enumerate(_PRIMES_UP_TO_40):
        while n % p == 0:
            vec[i] += 1
            n //= p
    return tuple(vec)

# Precompute for all candidates
_EXP_VECTORS = {x: _prime_exp_vector(x) for x in range(2, 41)}

def _vectors_conflict(v1, v2):
    """
    Check if two prime exponent vectors are linearly dependent over rationals.
    Two vectors are linearly dependent if one is a scalar multiple of the other.
    For non-negative integer vectors, this means v1 = c * v2 for some rational c > 0.
    Equivalently, for all i,j: v1[i]*v2[j] == v1[j]*v2[i].
    """
    # Find first non-zero component in each
    nz1 = -1
    nz2 = -1
    for i in range(_NUM_PRIMES):
        if v1[i] != 0:
            nz1 = i
            break
    for i in range(_NUM_PRIMES):
        if v2[i] != 0:
            nz2 = i
            break
    
    if nz1 == -1 or nz2 == -1:
        return False  # One is zero vector (shouldn't happen for n>=2)
    
    # Check if v1 = (v1[nz1]/v2[nz2]) * v2
    a = v1[nz1]
    b = v2[nz2]
    # Need b * v1[i] == a * v2[i] for all i
    for i in range(_NUM_PRIMES):
        if b * v1[i] != a * v2[i]:
            return False
    return True

def _can_add(products, x):
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

def _num_prime_factors(x):
    count = 0
    d = 2
    n = x
    while d * d <= n:
        while n % d == 0:
            count += 1
            n //= d
        d += 1
    if n > 1:
        count += 1
    return count

def _conflict_degree(x, all_candidates):
    """Count how many other candidates conflict with x."""
    vx = _EXP_VECTORS[x]
    count = 0
    for y in all_candidates:
        if x == y:
            continue
        if _vectors_conflict(vx, _EXP_VECTORS[y]):
            count += 1
    return count

def _greedy_construct(candidates):
    """Greedy: add candidates in given order if possible."""
    products = {1}
    selected = []
    for x in candidates:
        if time.time() > _DEADLINE:
            break
        if _can_add(products, x):
            products = _add_number(products, x)
            selected.append(x)
    return selected

def _backtrack_search(candidates, products, selected, depth, best):
    """Backtracking search to find larger sets with improved pruning."""
    if time.time() > _DEADLINE:
        return
    
    if len(selected) > len(best[0]):
        best[0] = list(selected)
    
    if depth >= len(candidates):
        return
    
    # Pruning: if we can't beat the best size with remaining candidates
    if len(selected) + (len(candidates) - depth) <= len(best[0]):
        return
    
    x = candidates[depth]
    
    # Try including x
    if _can_add(products, x):
        new_prods = _add_number(products, x)
        selected.append(x)
        _backtrack_search(candidates, new_prods, selected, depth + 1, best)
        selected.pop()
    
    # Try excluding x
    _backtrack_search(candidates, products, selected, depth + 1, best)

def _local_search(initial_set, candidates):
    """Try to improve by adding elements or swapping."""
    if time.time() > _DEADLINE:
        return list(initial_set)
    
    current = list(initial_set)
    current_prods = {1}
    for x in current:
        current_prods = _add_number(current_prods, x)
    
    improved = True
    while improved and time.time() < _DEADLINE:
        improved = False
        
        # First try adding elements
        for x in candidates:
            if time.time() > _DEADLINE:
                break
            if x in current:
                continue
            if _can_add(current_prods, x):
                current.append(x)
                current_prods = _add_number(current_prods, x)
                improved = True
        
        # Then try swapping: remove one, add another
        if not improved:
            for i in range(len(current)):
                if time.time() > _DEADLINE:
                    break
                remaining = current[:i] + current[i+1:]
                test_prods = {1}
                for x in remaining:
                    test_prods = _add_number(test_prods, x)
                
                for x in candidates:
                    if time.time() > _DEADLINE:
                        break
                    if x in current:
                        continue
                    if _can_add(test_prods, x):
                        current[i] = x
                        current_prods = _add_number(test_prods, x)
                        improved = True
                        break
                if improved:
                    break
    
    return current

def _verify(nums):
    """Verify that all subset products are distinct."""
    if not nums:
        return True
    products = {1}
    for x in nums:
        new_prods = set()
        for p in products:
            new_prods.add(p * x)
        if new_prods & products:
            return False
        products = products | new_prods
    return len(products) == (1 << len(nums))

def construct():
    """
    Find the largest set of distinct integers in 2..40 such that all
    2^k subset products (including empty product = 1) are distinct.
    """
    candidates = list(range(2, 41))
    
    # Separate primes and composites
    primes = [x for x in candidates if _is_prime(x)]
    composites = [x for x in candidates if not _is_prime(x)]
    
    # Order composites by conflict degree (fewer conflicts first)
    composites_by_conflict = sorted(composites, key=lambda x: _conflict_degree(x, candidates))
    
    # Multiple greedy orderings
    orders = [
        primes + composites_by_conflict,
        primes + sorted(composites, key=lambda x: _num_prime_factors(x)),
        primes + sorted(composites, key=lambda x: -_num_prime_factors(x)),
        primes + composites,
        composites_by_conflict + primes,
        sorted(candidates, key=lambda x: _conflict_degree(x, candidates)),
        sorted(candidates, key=lambda x: x),
        sorted(candidates, key=lambda x: -x),
    ]
    
    best_set = []
    for order in orders:
        if time.time() > _DEADLINE:
            break
        result = _greedy_construct(order)
        if len(result) > len(best_set):
            best_set = result
    
    # Backtracking search with time limit
    if time.time() < _DEADLINE - 5:
        # Order: primes first (always compatible), then composites by conflict degree
        search_candidates = primes + composites_by_conflict
        best = [best_set]
        _backtrack_search(search_candidates, {1}, [], 0, best)
        if len(best[0]) > len(best_set):
            best_set = best[0]
    
    # Local search to improve
    if time.time() < _DEADLINE - 2:
        improved = _local_search(best_set, candidates)
        if len(improved) > len(best_set):
            best_set = improved
    
    # Final verification
    if not _verify(best_set):
        best_set = _greedy_construct(primes + composites_by_conflict)
    
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
