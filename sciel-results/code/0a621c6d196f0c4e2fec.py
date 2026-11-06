"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import sys

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    MAX_N = 10**12
    
    def sieve_primes(limit):
        """Generate primes up to limit using sieve."""
        if limit < 2:
            return []
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Precompute small primes for trial division
    SMALL_PRIME_LIMIT = 200000
    small_primes = sieve_primes(SMALL_PRIME_LIMIT)
    
    def is_prime(n):
        """Deterministic Miller-Rabin for n < 3.3e24."""
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = x * x % n
                if x == n - 1:
                    break
            else:
                return False
        return True
    
    def phi(x):
        """Compute Euler's totient function phi(x) exactly."""
        if x == 1:
            return 1
        result = x
        y = x
        for p in small_primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result
    
    def batch_phi(start, length):
        """Compute phi for consecutive integers [start, start+length-1] using segmented sieve.
        Returns list of phi values."""
        if length <= 0:
            return []
        
        # For small ranges or when start is small, use individual phi
        if start < 10**6 and length < 10000:
            return [phi(start + i) for i in range(length)]
        
        # Segmented sieve approach for larger ranges
        upper = start + length - 1
        sqrt_upper = int(math.isqrt(upper)) + 1
        
        # Get primes up to sqrt_upper
        if sqrt_upper <= SMALL_PRIME_LIMIT:
            primes_needed = small_primes
        else:
            is_prime_arr = bytearray(b'\x01') * (sqrt_upper + 1)
            is_prime_arr[0] = is_prime_arr[1] = 0
            for i in range(2, int(math.isqrt(sqrt_upper)) + 1):
                if is_prime_arr[i]:
                    is_prime_arr[i*i::i] = bytearray(len(is_prime_arr[i*i::i]))
            primes_needed = [i for i in range(2, sqrt_upper + 1) if is_prime_arr[i]]
        
        # Initialize: rem[i] = start + i (the remaining part after dividing out small prime factors)
        # phi_val[i] = start + i (will be modified to compute phi)
        rem = list(range(start, start + length))
        phi_val = list(range(start, start + length))
        
        for p in primes_needed:
            if p * p > upper:
                break
            first = ((start + p - 1) // p) * p
            for j in range(first, start + length, p):
                idx = j - start
                if rem[idx] % p == 0:
                    phi_val[idx] -= phi_val[idx] // p
                    while rem[idx] % p == 0:
                        rem[idx] //= p
        
        # Handle remaining prime factors
        for i in range(length):
            n = start + i
            if rem[i] > 1:
                phi_val[i] -= phi_val[i] // rem[i]
            if n == 1:
                phi_val[i] = 1
        
        return phi_val
    
    def check_run_fast(start, max_length, max_n):
        """Check consecutive integers from start for distinct phi values using batch computation.
        Returns length of longest run starting at start."""
        if start < 1:
            return 0
        if start + max_length - 1 > max_n:
            max_length = max_n - start + 1
            if max_length <= 0:
                return 0
        
        phi_vals = batch_phi(start, max_length)
        
        seen = set()
        k = 0
        for v in phi_vals:
            if v in seen:
                break
            seen.add(v)
            k += 1
        return k
    
    best_k = 1
    best_s = 1
    
    # Phase 1: Dense search in small ranges
    search_limit = 200000
    for s in range(1, min(search_limit, int(MAX_N)) + 1):
        if time.time() > deadline - 10:
            break
        k = check_run_fast(s, 200, MAX_N)
        if k > best_k:
            best_k = k
            best_s = s
    
    # Phase 2: Medium-range search
    if time.time() < deadline - 20:
        for s in range(200000, min(2000000, int(MAX_N)) + 1, 10):
            if time.time() > deadline - 10:
                break
            k = check_run_fast(s, 200, MAX_N)
            if k > best_k:
                best_k = k
                best_s = s
    
    # Phase 3: Targeted search around primorials
    if time.time() < deadline - 20:
        primorial = 1
        prime_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        for p in prime_list:
            primorial *= p
            if primorial > MAX_N:
                break
            for offset in range(-500, 501, 10):
                if time.time() > deadline - 10:
                    break
                s = primorial + offset
                if s < 1 or s + 199 > MAX_N:
                    continue
                k = check_run_fast(s, 200, MAX_N)
                if k > best_k:
                    best_k = k
                    best_s = s
    
    # Phase 4: Random sampling in larger ranges
    if time.time() < deadline - 20:
        ranges = [
            (10**6, 10**7),
            (10**7, 10**8),
            (10**8, 10**9),
            (10**9, 10**10),
            (10**10, 10**11),
            (10**11, 10**12),
        ]
        for lo, hi in ranges:
            if time.time() > deadline - 10:
                break
            num_samples = 500
            for _ in range(num_samples):
                if time.time() > deadline - 10:
                    break
                s = random.randint(lo, min(hi, int(MAX_N) - 200))
                k = check_run_fast(s, 200, MAX_N)
                if k > best_k:
                    best_k = k
                    best_s = s
    
    # Phase 5: Local refinement
    if best_k > 5 and time.time() < deadline - 10:
        window = 2000
        for offset in range(-window, window + 1, 2):
            if time.time() > deadline - 10:
                break
            s = best_s + offset
            if s < 1 or s + 199 > MAX_N:
                continue
            k = check_run_fast(s, 200, MAX_N)
            if k > best_k:
                best_k = k
                best_s = s
    
    # Phase 6: Final fine-grained check
    if time.time() < deadline - 5:
        fine_window = 200
        for offset in range(-fine_window, fine_window + 1):
            if time.time() > deadline - 2:
                break
            s = best_s + offset
            if s < 1 or s + 199 > MAX_N:
                continue
            k = check_run_fast(s, 200, MAX_N)
            if k > best_k:
                best_k = k
                best_s = s
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
