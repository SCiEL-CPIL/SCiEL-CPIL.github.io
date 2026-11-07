import time
import math
from fractions import Fraction

# EVOLVE-BLOCK-START

def _is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.isqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def _max_powers_with_distinct_sums(p, limit):
    """
    Find the largest subset of powers p^1, p^2, ... p^k <= limit
    such that the exponents form a set with distinct subset sums.
    """
    powers = []
    exp = 1
    val = p
    while val <= limit:
        powers.append((exp, val))
        exp += 1
        val *= p
    
    # Greedy selection of exponents
    # We want to select exponents e1, e2, ... such that subset sums are distinct.
    # This is satisfied if e_i > sum(e_1...e_{i-1}).
    selected_exponents = []
    current_sum = 0
    for exp, val in powers:
        if exp > current_sum:
            selected_exponents.append((exp, val))
            current_sum += exp
    
    return [val for _, val in selected_exponents]

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    limit = 40
    primes = [n for n in range(2, limit + 1) if _is_prime(n)]
    
    result = []
    for p in primes:
        # Add max powers of p
        powers = _max_powers_with_distinct_sums(p, limit)
        result.extend(powers)
        
    # The result should now contain all selected prime powers.
    # We should verify that the subset products are distinct.
    # For small sets, we can just check.
    
    # Sort result for determinism
    result.sort()
    
    # Verification
    # If the set is large, 2^k might be big.
    # k=16, 2^16 = 65536. This is fine.
    def _check_distinct_products(nums):
        products = set()
        k = len(nums)
        for mask in range(1 << k):
            prod = 1
            for i in range(k):
                if mask & (1 << i):
                    prod *= nums[i]
            if prod in products:
                return False
            products.add(prod)
        return True

    if not _check_distinct_products(result):
        # Fallback to primes only if the greedy construction failed (shouldn't happen)
        result = primes
        # But primes are 12, so score 1.0.
        # We expect 16.
        
    # Ensure types are Python int
    result = [int(x) for x in result]
    
    return result

# EVOLVE-BLOCK-END
