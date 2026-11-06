import time
import sys
import numpy as np
from math import isqrt
from scipy.signal import fftconvolve

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
    max_b = int(limit ** (1.0 / 3.0)) + 2

    # Sieve of Eratosthenes up to max_b to get primes
    max_b_plus = max_b + 1
    if max_b_plus > 1:
        is_prime = bytearray(b'\x01') * (max_b_plus)
        is_prime[0] = 0
        if max_b_plus > 1:
            is_prime[1] = 0
        for i in range(2, isqrt(max_b) + 1):
            if is_prime[i]:
                for j in range(i * i, max_b_plus, i):
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
    deadline = start_time + 130  # 20s safety margin before 150s limit

    best_result = {'N': 5, 'pairs': [[1, 4]]}
    best_score = 1

    # Try FFT-based convolution with different N_MAX values
    # N_MAX determines the range of sums we search
    # FFT array size = 2 * N_MAX
    # For N_MAX=20M: array 40M, FFT ~640MB (float32)
    # For N_MAX=50M: array 100M, FFT ~1.6GB
    # For N_MAX=100M: array 200M, FFT ~3.2GB (may be too much)
    
    N_MAX_candidates = [20_000_000, 40_000_000, 60_000_000, 100_000_000]

    for N_MAX in N_MAX_candidates:
        if time.time() > deadline - 25:
            break
        
        pow_limit = N_MAX
        try:
            powerful = _generate_powerful_numbers(pow_limit)
        except Exception:
            continue
        
        K = len(powerful)
        if K < 2:
            continue
        
        elapsed_gen = time.time() - start_time
        if elapsed_gen > deadline - 30:
            break
        
        # Create indicator array
        arr_size = N_MAX + 1
        try:
            arr = np.zeros(arr_size, dtype=np.float32)
        except MemoryError:
            continue
        
        for x in powerful:
            if x <= N_MAX:
                arr[x] = 1.0
        
        # FFT convolution
        try:
            conv = fftconvolve(arr, arr, mode='full')
        except Exception:
            del arr
            continue
        
        del arr
        
        # conv[n] = number of ordered pairs (a,b) with a+b=n where a,b are powerful
        # We want a <= b, so unordered count = (conv[n] + (1 if n even and arr[n//2]>0 else 0)) / 2
        # But for finding max, we can just use conv[n] directly
        
        # Valid N values: 2 to 2*N_MAX (since a,b >= 1, min sum = 2)
        # conv has length 2*N_MAX + 1 (indices 0 to 2*N_MAX)
        if len(conv) < 3:
            del conv
            continue
        
        # Find best N (skip indices 0,1)
        conv_valid = conv[2:]
        if len(conv_valid) == 0:
            del conv
            continue
        
        best_idx = int(np.argmax(conv_valid)) + 2
        best_conv_val = int(conv[best_idx])
        
        # Compute unordered pair count
        # Need to check if best_idx//2 is powerful
        half = best_idx // 2
        if best_idx % 2 == 0 and half <= N_MAX:
            # Check if half is powerful
            is_half_pow = False
            for x in powerful:
                if x == half:
                    is_half_pow = True
                    break
            if is_half_pow:
                unordered_count = (best_conv_val + 1) // 2
            else:
                unordered_count = best_conv_val // 2
        else:
            unordered_count = best_conv_val // 2
        
        del conv
        
        if unordered_count > best_score:
            # Extract pairs
            pow_set = set(powerful)
            pairs = []
            half_N = best_idx // 2
            for a in powerful:
                if a > half_N:
                    break
                b = best_idx - a
                if b in pow_set:
                    if _is_powerful(a) and _is_powerful(b):
                        pairs.append([a, b])
            
            if len(pairs) > best_score:
                best_score = len(pairs)
                best_result = {'N': best_idx, 'pairs': pairs}
        
        del powerful
    
    if best_score <= 1:
        return {'N': 5, 'pairs': [[1, 4]]}
    
    return best_result
