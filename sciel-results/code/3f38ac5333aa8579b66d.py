"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def _prime_factors(n):
    """Return set of prime factors of n."""
    factors = set()
    if n <= 1:
        return factors
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _count_distinct_primes(nums):
    """Count distinct primes dividing all pairwise sums of distinct elements."""
    all_primes = set()
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            s = nums[i] + nums[j]
            all_primes |= _prime_factors(s)
    return len(all_primes)

def _evaluate_exponents(exponents):
    """Evaluate a set of exponents for powers of 2."""
    nums = [2 ** e for e in exponents]
    if max(nums) > 10000:
        return float('inf')
    return _count_distinct_primes(nums)

def construct():
    deadline = time.time() + 80  # safety margin before 90s limit
    
    # Strategy: use powers of 2 with carefully chosen exponents
    # All sums will be 2^min(a,b) * (1 + 2^|a-b|)
    # Distinct primes come from odd parts (1 + 2^k) for k in difference set
    
    # Try consecutive exponents starting from 0
    # Max value: 2^11 = 2048 <= 10000, so exponents 0-11 work
    base_exponents = list(range(12))
    
    best_score = _evaluate_exponents(base_exponents)
    best_exponents = base_exponents[:]
    
    # Try shifting the exponent range
    for start in range(0, 5):
        exponents = [start + i for i in range(12)]
        if 2 ** max(exponents) > 10000:
            break
        score = _evaluate_exponents(exponents)
        if score < best_score:
            best_score = score
            best_exponents = exponents[:]
        if time.time() > deadline:
            break
    
    # Try non-consecutive exponent sets to minimize distinct differences
    # Key: minimize the number of distinct |a-b| values
    # With 12 points, we want differences to fall into few distinct values
    
    # Try arithmetic progressions with various step sizes
    for step in range(1, 4):
        for start in range(0, 5):
            exponents = [start + i * step for i in range(12)]
            if 2 ** max(exponents) > 10000:
                continue
            score = _evaluate_exponents(exponents)
            if score < best_score:
                best_score = score
                best_exponents = exponents[:]
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Try mixed strategies: some powers of 2, possibly scaled
    # If all numbers are 2^k * m where m is a fixed odd number,
    # then sums are 2^k * m * (2^(a-b) + 1) type structures
    
    # Try: all numbers are multiples of a small odd number times powers of 2
    # e.g., nums[i] = odd_base * 2^(e_i)
    for odd_base in [3, 5, 7]:
        for start in range(0, 4):
            exponents = [start + i for i in range(12)]
            nums = [odd_base * (2 ** e) for e in exponents]
            if max(nums) > 10000:
                continue
            if len(set(nums)) != 12:
                continue
            score = _count_distinct_primes(nums)
            if score < best_score:
                best_score = score
                best_exponents = exponents[:]
                # Store the odd_base for final construction
                best_odd_base = odd_base
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Final: construct the result
    # Check if we found a better solution with odd_base
    if 'best_odd_base' in dir() and best_odd_base is not None:
        # Re-evaluate to confirm
        pass
    
    # Use the best exponents found
    nums = [2 ** e for e in best_exponents]
    
    # Verify constraints
    assert len(nums) == 12
    assert len(set(nums)) == 12
    assert all(1 <= x <= 10000 for x in nums)
    
    return nums
# EVOLVE-BLOCK-END
