"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 40, 'maximum_size': 18}

def _get_prime_powers(n):
    """Get all prime powers in [2, n]."""
    prime_powers = set()
    for a in range(2, n + 1):
        # Check if a is a prime power: a = p^k for some prime p and k >= 1
        temp = a
        p = None
        while temp % 2 == 0:
            p = 2
            temp //= 2
        if temp == 1:
            prime_powers.add(a)
            continue
        for d in range(3, int(temp**0.5) + 1, 2):
            if temp % d == 0:
                p = d
                while temp % d == 0:
                    temp //= d
                break
        if temp == 1:
            prime_powers.add(a)
    return sorted(prime_powers)

def _has_distinct_subset_products(nums):
    """Check if all subset products are distinct."""
    products = set()
    n = len(nums)
    for mask in range(1 << n):
        prod = 1
        for i in range(n):
            if mask & (1 << i):
                prod *= nums[i]
        if prod in products:
            return False
        products.add(prod)
    return True

def _build_max_set(candidates, deadline):
    """Greedy repair: start with all candidates, remove elements causing collisions."""
    current = list(candidates)
    
    while len(current) > 1:
        if time.time() > deadline:
            break
        if _has_distinct_subset_products(current):
            break
        
        # Find which element to remove: try removing each and see if it helps
        best_remove_idx = None
        best_score = -1
        
        for i in range(len(current)):
            candidate = current[:i] + current[i+1:]
            if _has_distinct_subset_products(candidate):
                best_remove_idx = i
                break
        
        if best_remove_idx is None:
            # No single removal fixes it; remove the largest element as heuristic
            best_remove_idx = max(range(len(current)), key=lambda i: current[i])
        
        current.pop(best_remove_idx)
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    # Get prime powers in 2..40
    candidates = _get_prime_powers(P['n'])
    
    # Greedy repair
    result = _build_max_set(candidates, deadline)
    
    # Ensure we return at least 1 element
    if not result:
        result = [2]
    
    # Verify final result
    if not _has_distinct_subset_products(result):
        # Fallback to primes
        from math import isqrt
        result = [a for a in range(2, P['n'] + 1) if all(a % d for d in range(2, isqrt(a) + 1))]
    
    return list(result)
# EVOLVE-BLOCK-END
