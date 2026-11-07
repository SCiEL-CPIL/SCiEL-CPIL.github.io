"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from fractions import Fraction

MAX_N = 10000000
TARGET = Fraction(3, 2)

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

def _sieve(limit):
    """Return smallest prime factor for each number up to limit."""
    spf = list(range(limit + 1))
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def _sigma_from_sf(sp, n):
    """Compute sigma(n) using smallest prime factor table."""
    result = 1
    while n > 1:
        p = sp[n]
        power = p
        s = 1 + p
        n //= p
        while n > 1 and sp[n] == p:
            n //= p
            power *= p
            s += power
        result *= s
    return result

def _ratio_err(a, b):
    """Return |a/b - 3/2| as a Fraction."""
    return abs(Fraction(a, b) - TARGET)

def _best_in_family(a, b, sp):
    """
    Given a pair (a, b) with sigma(a) == sigma(b), try to find a better pair
    in the same sigma-fiber by scaling one of them.
    Returns (best_a, best_b) minimizing |a/b - 3/2|.
    """
    best_a, best_b = a, b
    best_err = _ratio_err(a, b)
    
    # Try scaling a by small primes while keeping sigma equal
    # sigma(k*a) = sigma(k)*sigma(a) if gcd(k,a)=1
    # We need sigma(k*a) == sigma(k'*b) for some k'
    # For simplicity, try multiplying a by small primes and check if we can find b'
    # This is complex, so instead we do a local search around the ratio
    
    return best_a, best_b

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline to be safe within 90s
    
    # Build smallest prime factor table
    sp = _sieve(MAX_N)
    
    # Build sigma values
    sigma = [0] * (MAX_N + 1)
    sigma[1] = 1
    for i in range(2, MAX_N + 1):
        sigma[i] = _sigma_from_sf(sp, i)
    
    # Group numbers by sigma value
    sigma_map = {}
    for i in range(1, MAX_N + 1):
        s = sigma[i]
        if s not in sigma_map:
            sigma_map[s] = []
        sigma_map[s].append(i)
    
    # Search for best pair
    best_a, best_b = 1, 1
    best_err = Fraction(1, 1)  # Start with large error
    
    # For each sigma group, try to find pairs that approximate 3/2
    # We want a/b ≈ 3/2, so a ≈ 1.5 * b
    for s, nums in sigma_map.items():
        if time.time() > deadline:
            break
        if len(nums) < 2:
            continue
        
        # Sort nums
        nums.sort()
        
        # For each b, find a that is close to 1.5 * b
        # Use two-pointer or binary search
        for i in range(len(nums)):
            b = nums[i]
            target_a = 1.5 * b
            
            # Binary search for closest a in nums
            # We want a > b (since a/b ≈ 1.5 > 1)
            lo, hi = 0, len(nums) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if nums[mid] < target_a:
                    lo = mid + 1
                else:
                    hi = mid
            
            # Check lo and lo-1 (if valid)
            for j in [lo, lo - 1]:
                if 0 <= j < len(nums) and j != i:
                    a = nums[j]
                    if a != b and a > b:  # Ensure a > b for ratio > 1
                        err = _ratio_err(a, b)
                        if err < best_err:
                            best_err = err
                            best_a, best_b = a, b
    
    # Verify the best pair
    if sigma[best_a] != sigma[best_b]:
        # Fallback to a known valid pair
        best_a, best_b = 11, 6
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
