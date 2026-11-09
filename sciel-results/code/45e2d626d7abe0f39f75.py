"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt, gcd
from itertools import combinations

def _sieve_primes(n):
    if n < 2:
        return []
    is_p = [True] * (n + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, isqrt(n) + 1):
        if is_p[i]:
            for j in range(i * i, n + 1, i):
                is_p[j] = False
    return [i for i in range(2, n + 1) if is_p[i]]

def _factorize(x, primes):
    exponents = []
    for p in primes:
        e = 0
        while x % p == 0:
            x //= p
            e += 1
        exponents.append(e)
    return exponents

def _check_valid_subset(nums):
    """Check if all subset products are distinct by direct enumeration."""
    if len(nums) == 0:
        return True
    k = len(nums)
    if k > 20:
        # Too many to enumerate directly; use linear independence as necessary condition
        return _check_q_independent(nums)
    products = set()
    for mask in range(1 << k):
        prod = 1
        for i in range(k):
            if mask & (1 << i):
                prod *= nums[i]
        if prod in products:
            return False
        products.add(prod)
    return True

def _check_q_independent(nums):
    """Check Q-linear independence of prime exponent vectors."""
    n = 40
    primes = _sieve_primes(n)
    vecs = []
    for x in nums:
        vecs.append(_factorize(x, primes))
    if not vecs:
        return True
    # Gaussian elimination over Q using exact fractions
    ncols = len(primes)
    nrows = len(vecs)
    mat = [row[:] for row in vecs]
    rank = 0
    for col in range(ncols):
        pivot_row = -1
        for r in range(rank, nrows):
            if mat[r][col] != 0:
                pivot_row = r
                break
        if pivot_row == -1:
            continue
        mat[rank], mat[pivot_row] = mat[pivot_row], mat[rank]
        pivot_val = mat[rank][col]
        for r in range(rank + 1, nrows):
            factor = mat[r][col]
            if factor != 0:
                for c in range(col, ncols):
                    mat[r][c] = mat[r][c] * pivot_val - factor * mat[rank][c]
                g = 0
                for c in range(col, ncols):
                    g = gcd(g, abs(mat[r][c]))
                if g > 1:
                    for c in range(col, ncols):
                        mat[r][c] //= g
        rank += 1
        if rank == nrows:
            return True
    return rank == nrows

def _construct():
    t0 = time.time()
    deadline = t0 + 55.0
    
    n = 40
    primes = _sieve_primes(n)
    all_vecs = {}
    for x in range(2, n + 1):
        all_vecs[x] = _factorize(x, primes)
    
    # Start with all 12 primes as baseline
    prime_nums = [x for x in range(2, 41) if all(x % d != 0 for d in range(2, isqrt(x) + 1))]
    
    # Try to add composites one by one
    # Strategy: greedy with backtracking
    # First, get a good initial set using greedy
    candidates = list(range(2, 41))
    
    # Try different orderings to find a large valid set
    best_set = list(prime_nums)
    best_size = len(best_set)
    
    # Greedy: try adding each composite to the prime set
    composites = [x for x in range(2, 41) if x not in prime_nums]
    
    # Try multiple greedy strategies
    strategies = [
        sorted(composites, key=lambda x: -x),  # largest first
        sorted(composites, key=lambda x: x),   # smallest first
        sorted(composites, key=lambda x: sum(all_vecs[x])),  # fewest prime factors first
        sorted(composites, key=lambda x: -sum(all_vecs[x])),  # most prime factors first
    ]
    
    for ordering in strategies:
        current = list(prime_nums)
        for c in ordering:
            if time.time() > deadline - 5:
                break
            trial = current + [c]
            if _check_valid_subset(trial):
                current = trial
        if len(current) > best_size:
            best_size = len(current)
            best_set = current
    
    # Now try to improve by swapping: remove one element, add others
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        current = list(best_set)
        for remove_idx in range(len(current)):
            if time.time() > deadline - 5:
                break
            remaining = [x for i, x in enumerate(current) if i != remove_idx]
            removed = current[remove_idx]
            # Try adding composites not in current
            for c in range(2, 41):
                if c in current or c == removed:
                    continue
                trial = remaining + [c]
                if _check_valid_subset(trial):
                    if len(trial) > len(best_set):
                        best_set = trial
                        best_size = len(trial)
                        improved = True
                        break
    
    # Final verification
    assert len(best_set) >= 12
    assert all(isinstance(x, int) for x in best_set)
    assert all(2 <= x <= 40 for x in best_set)
    assert len(set(best_set)) == len(best_set)
    assert _check_valid_subset(best_set)
    
    return list(best_set)

def construct():
    return _construct()
# EVOLVE-BLOCK-END
