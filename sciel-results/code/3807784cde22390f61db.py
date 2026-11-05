import time
import sys
import numpy as np
from math import isqrt

def _is_powerful(n):
    """Check if n is a powerful number (every prime factor appears with exponent >= 2)."""
    if n < 1:
        return False
    if n == 1:
        return True
    temp = n
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97)
    for p in small_primes:
        pp = p * p
        if pp > temp:
            break
        if temp % pp == 0:
            while temp % p == 0:
                temp //= p
        elif temp % p == 0:
            return False
    if temp > 1:
        r = isqrt(temp)
        if r * r != temp:
            return False
    return True

def _generate_powerful_numbers(limit):
    """Generate all powerful numbers up to limit using n = a^2 * b^3, b square-free."""
    powerful = set()
    powerful.add(1)
    max_b = int(limit ** (1.0/3.0)) + 2
    
    # Sieve of Eratosthenes up to max_b to get primes
    max_b_plus = max_b + 1
    if max_b_plus > 1:
        is_prime = bytearray(b'\x01') * (max_b_plus)
        is_prime[0] = 0
        if max_b_plus > 1:
            is_prime[1] = 0
        for i in range(2, isqrt(max_b) + 1):
            if is_prime[i]:
                for j in range(i*i, max_b_plus, i):
                    is_prime[j] = 0
        small_primes = [i for i in range(2, max_b_plus) if is_prime[i]]
    else:
        small_primes = []
    
    for b in range(1, max_b + 1):
        # Check if b is square-free
        t = b
        is_sf = True
        for p in small_primes:
            if p * p > t:
                break
            if t % (p * p) == 0:
                is_sf = False
                break
        if not is_sf:
            continue
        
        b3 = b * b * b
        if b3 > limit:
            continue
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
            else:
                break
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    # We will use Numpy to vectorize the sum counting.
    # This allows us to handle much larger K (number of powerful numbers)
    # compared to the pure Python loop.
    
    # Estimate K for a given limit L: K ~ 2.16 * sqrt(L)
    # If L = 10^8, K ~ 21600. K^2 ~ 4.6 * 10^8 ops.
    # If L = 10^9, K ~ 68000. K^2 ~ 4.6 * 10^9 ops.
    # Numpy can handle ~100-500M ops/sec for simple arithmetic + bincount.
    # 4.6B ops might take 10-50 seconds.
    # Let's aim for L = 10^8 first, and if time permits, we could go higher, 
    # but we generate once. Let's try L = 2 * 10^8.
    
    pow_limit = 200_000_000  # 2 * 10^8
    
    powerful = _generate_powerful_numbers(pow_limit)
    K = len(powerful)
    
    if time.time() > deadline - 60:
        # Fallback to smaller limit if generation was unexpectedly slow
        pow_limit = 50_000_000
        powerful = _generate_powerful_numbers(pow_limit)
        K = len(powerful)

    # Convert to numpy array for fast operations
    # Use int64 to be safe with sums up to 2 * 10^8
    pow_arr = np.array(powerful, dtype=np.int64)
    
    # We need to count sums a+b for all a,b in pow_arr.
    # Max sum is 2 * pow_limit.
    max_sum = 2 * pow_limit + 1
    counts = np.zeros(max_sum, dtype=np.int32)
    
    # Chunked processing to manage memory
    # Memory for one chunk of size C: K * C * 8 bytes
    # If K=50000, C=1000, Mem = 50000 * 1000 * 8 = 400 MB. OK.
    chunk_size = 1000
    
    for i in range(0, K, chunk_size):
        chunk = pow_arr[i:i+chunk_size]
        # Compute outer sum: shape (K, chunk_size)
        # This creates a temporary array of size K * chunk_size
        sums = pow_arr[:, None] + chunk[None, :]
        
        # Flatten and bincount
        # minlength ensures we cover up to max_sum
        chunk_counts = np.bincount(sums.ravel(), minlength=max_sum)
        counts += chunk_counts
        
        # Time check
        if time.time() > deadline - 15:
            break

    if not any(counts > 0):
        return {'N': 5, 'pairs': [[1, 4]]}

    # Find the best N
    best_N = int(np.argmax(counts))
    best_count = int(counts[best_N])
    
    # Extract pairs
    # For efficiency, convert powerful to a set for O(1) lookup
    pow_set = set(powerful)
    pairs = []
    
    # We only need a <= b, so a <= best_N / 2
    half_N = best_N // 2
    for a in powerful:
        if a > half_N:
            break
        b = best_N - a
        if b in pow_set:
            # Verify powerful property just in case (though generation is correct)
            # _is_powerful is slow, but we only do this for pairs of the best N
            # Number of pairs is small (likely < 100)
            if _is_powerful(a) and _is_powerful(b):
                pairs.append([a, b])
    
    if not pairs:
        # Fallback
        return {'N': 5, 'pairs': [[1, 4]]}
        
    return {'N': best_N, 'pairs': pairs}
