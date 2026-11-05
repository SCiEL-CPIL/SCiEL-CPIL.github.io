import math
import time
import numpy as np

def _get_primes(limit):
    """Sieve of Eratosthenes to get primes up to limit."""
    if limit < 2:
        return []
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i, val in enumerate(is_prime) if val]

# Precompute primes up to 10^6. sqrt(10^12) = 10^6.
# This is fast and done once.
PRIMES = _get_primes(10**6)

def construct():
    start_time = time.time()
    LIMIT = 110.0  # Safety margin for 120s limit
    MAX_S = 10**12
    MAX_K = 200
    
    primes = PRIMES
    primes_len = len(primes)
    
    def batch_phi(nums):
        """
        Compute phi for a list of numbers using vectorized operations.
        nums: list of integers
        returns: list of phi values
        """
        n = len(nums)
        # Use int64 to avoid overflow? 10^12 fits in int64.
        # But phi operations involve division.
        # We need to be careful with numpy int types.
        # Python integers are arbitrary precision, but numpy int64 is fixed.
        # 10^12 is safe for int64.
        m = np.array(nums, dtype=np.int64)
        result = m.copy()
        
        # Check 2
        mask = (m % 2 == 0)
        if np.any(mask):
            result[mask] -= result[mask] // 2
            while np.any(m[mask] % 2 == 0):
                m[mask] //= 2
        
        # Check odd primes
        for i in range(1, primes_len):
            p = primes[i]
            if p * p > np.max(m):
                break
            
            mask = (m % p == 0)
            if not np.any(mask):
                continue
            
            result[mask] -= result[mask] // p
            while np.any(m[mask] % p == 0):
                m[mask] //= p
        
        # Remaining factors
        mask = (m > 1)
        if np.any(mask):
            result[mask] -= result[mask] // m[mask]
            
        return result.tolist()

    def is_valid_window(s):
        """
        Check if phi(s) ... phi(s+MAX_K-1) are distinct.
        """
        if s + MAX_K - 1 > MAX_S:
            return False
            
        # Generate the window
        window = list(range(s, s + MAX_K))
        phis = batch_phi(window)
        
        # Check distinctness
        return len(set(phis)) == MAX_K

    # Strategy:
    # 1. Deterministic scan near 10^12
    # 2. Random search
    
    base_start = MAX_S - 1000
    
    for offset in range(0, 1000):
        if time.time() - start_time > LIMIT:
            break
            
        s = base_start + offset
        if is_valid_window(s):
            return {'start': s, 'length': MAX_K}

    # Fallback: Random search
    import random
    max_start = MAX_S - MAX_K + 1
    while time.time() - start_time < LIMIT:
        s = random.randint(1, max_start)
        if is_valid_window(s):
            return {'start': s, 'length': MAX_K}

    # Last resort
    return {'start': 1, 'length': 1}
