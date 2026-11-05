"""Rewritten program: minimize the largest distinct factor > 12 with product 12!."""
# EVOLVE-BLOCK-START
import math
import time
from functools import lru_cache

TARGET = math.factorial(12)
MIN_F = 13
MAX_FACTORS = 12
DEADLINE = 85.0  # safety margin before 90s kill


def _compute_divisors(n):
    """Compute all divisors of n."""
    divs = set([1])
    for p in range(2, int(math.isqrt(n)) + 1):
        if n % p == 0:
            while n % p == 0:
                n //= p
            new_divs = set()
            d = p
            while d <= n * p:
                for old_d in divs:
                    new_divs.add(old_d * d)
                d *= p
            divs |= new_divs
    return sorted(divs)


def _factorize(n):
    """Return prime factorization of n as dict {prime: exponent}."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        factors[n] = factors.get(n, 1)
    return factors


def construct():
    """Return 1..12 DISTINCT integers all greater than 12, with product exactly 12!.
    Minimize the LARGEST factor."""
    start = time.time()
    
    # Precompute divisors of TARGET
    all_divs = _compute_divisors(TARGET)
    
    # Filter: divisors > MIN_F
    valid_divs = [d for d in all_divs if d > MIN_F]
    valid_divs_set = set(valid_divs)
    
    # Initial solution: [TARGET]
    best_max = TARGET
    best_factors = [TARGET]
    
    # Use DFS with pruning
    # State: (remaining_product, min_factor_index, factors_left, current_max)
    # We build factors in increasing order to ensure distinctness
    
    def _lower_bound_product(min_idx, k):
        """Minimum possible product of k distinct valid divisors starting from min_idx."""
        if min_idx + k > len(valid_divs):
            return float('inf')
        prod = 1
        for i in range(k):
            prod *= valid_divs[min_idx + i]
            if prod > TARGET:
                return float('inf')
        return prod
    
    def _can_complete(remaining, min_idx, k):
        """Check if remaining can be factored into k distinct divisors >= valid_divs[min_idx]."""
        if k == 0:
            return remaining == 1
        if remaining < 1:
            return False
        if min_idx >= len(valid_divs):
            return False
        
        # Lower bound check
        lb = _lower_bound_product(min_idx, k)
        if lb > remaining:
            return False
        
        # Upper bound: remaining must be achievable
        # The largest factor we can use is limited by remaining
        if remaining < valid_divs[min_idx]:
            return False
        
        # Check if remaining itself is a valid divisor (for k=1)
        if k == 1:
            return remaining in valid_divs_set and remaining >= valid_divs[min_idx]
        
        return True
    
    def dfs(remaining, min_idx, k, current_max, current_factors):
        nonlocal best_max, best_factors
        
        if time.time() - start > DEADLINE:
            return
        
        if k == 0:
            if remaining == 1:
                if current_max < best_max:
                    best_max = current_max
                    best_factors = list(current_factors)
            return
        
        if remaining < 1:
            return
        
        if min_idx >= len(valid_divs):
            return
        
        # Pruning: lower bound
        lb = _lower_bound_product(min_idx, k)
        if lb > remaining:
            return
        
        if remaining < valid_divs[min_idx]:
            return
        
        # Try each possible next factor
        for i in range(min_idx, len(valid_divs)):
            f = valid_divs[i]
            
            if f > remaining:
                break
            
            if remaining % f != 0:
                continue
            
            # New max would be max(current_max, f)
            new_max = max(current_max, f)
            if new_max >= best_max:
                break  # Since f is increasing, all further f will also exceed best_max
            
            new_remaining = remaining // f
            new_min_idx = i + 1
            new_k = k - 1
            
            # Pruning: can we complete?
            if not _can_complete(new_remaining, new_min_idx, new_k):
                continue
            
            current_factors.append(f)
            dfs(new_remaining, new_min_idx, new_k, new_max, current_factors)
            current_factors.pop()
            
            # Early termination if we found a very good solution
            if best_max <= MIN_F + k + 5:
                return
    
    # Try different numbers of factors from 2 to MAX_FACTORS
    # Start with fewer factors (more likely to have small max)
    for k in range(2, MAX_FACTORS + 1):
        if time.time() - start > DEADLINE:
            break
        
        # Check if k factors is even possible
        if _lower_bound_product(0, k) > TARGET:
            break  # Can't fit k more factors
        
        dfs(TARGET, 0, k, MIN_F - 1, [])
        
        # If we found a solution with max very close to theoretical minimum, stop
        if best_max <= MIN_F + k + 3:
            break
    
    return best_factors


# EVOLVE-BLOCK-END
