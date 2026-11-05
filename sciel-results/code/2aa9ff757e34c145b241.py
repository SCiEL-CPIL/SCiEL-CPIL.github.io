import time
import random
import numpy as np
import itertools

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

def _paley_orientation(p):
    """Return orientation bits for a Paley tournament of prime order p (p ≡ 3 mod 4)."""
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    n = p
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % p
            if diff in squares:
                orientation.append(0) # i -> j
            else:
                orientation.append(1) # j -> i
    return {'n': n, 'orientation': orientation}

def _check_property_fast(n, orientation):
    """
    Check if every 4 vertices have a common dominator.
    Returns True if valid.
    Uses a vectorized-ish approach by iterating 4-sets and checking intersection of in-neighborhoods.
    """
    # Build in-neighborhood bitsets
    # in_neigh[u] is a bitmask of vertices v such that v -> u
    in_neigh = [0] * n
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orientation[idx]
            if bit == 0:
                # i -> j
                in_neigh[j] |= (1 << i)
            else:
                # j -> i
                in_neigh[i] |= (1 << j)
            idx += 1
            
    # Iterate all 4-sets
    # For n=67, C(67,4) is ~7.8M.
    # This loop is the bottleneck.
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        common = in_neigh[a] & in_neigh[b] & in_neigh[c] & in_neigh[d]
        mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
        if (common & ~mask) == 0:
            return False
    return True

def _check_property_numba_like(n, orientation):
    """
    Alternative check using numpy for potentially faster bit operations if possible,
    but python ints are usually faster for bit ops than numpy arrays for single values.
    This function is kept for potential future optimization or fallback, 
    but _check_property_fast is likely faster due to less overhead.
    """
    pass

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    # Primes p ≡ 3 mod 4
    # 67, 71, 79, 83
    primes_to_try = [67, 71, 79, 83]
    
    for p in primes_to_try:
        if time.time() - start_time > time_limit:
            break
        if p % 4 != 3 or not _is_prime(p):
            continue
            
        result = _paley_orientation(p)
        n = result['n']
        
        # Check if it satisfies the property
        if _check_property_fast(n, result['orientation']):
            return result
            
    # If none worked (should not happen for 83), return 83
    return _paley_orientation(83)
