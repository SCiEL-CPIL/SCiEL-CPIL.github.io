import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Leave 5s safety margin
    
    def now():
        return time.time()
    
    # --- Phase 1: NumPy Sieve for Small/Medium Range ---
    # Compute phi for all numbers up to N using sieve
    N = 10**7
    
    try:
        phi_arr = np.arange(N + 1, dtype=np.int64)
        
        # Standard totient sieve: for each prime p, update multiples
        for i in range(2, N + 1):
            if phi_arr[i] == i:  # i is prime
                phi_arr[i::i] -= phi_arr[i::i] // i
            if now() > DEADLINE - 15.0:
                break
        
        # Find max distinct run in phi_arr[1:]
        best_k = 1
        best_s = 1
        
        vals = phi_arr[1:]
        n_vals = len(vals)
        
        for i in range(n_vals):
            if n_vals - i < best_k:
                break
            seen = set()
            for j in range(i, min(i + 200, n_vals)):
                v = vals[j]
                if v in seen:
                    break
                seen.add(v)
                curr_k = j - i + 1
                if curr_k > best_k:
                    best_k = curr_k
                    best_s = i + 1
                    if best_k == 200:
                        break
            if best_k == 200:
                break
        
    except Exception:
        best_k = 1
        best_s = 1
    
    # --- Phase 2: Search in Larger Range with Individual Phi Computation ---
    # For numbers beyond N, use individual phi computation with Pollard's rho
    
    if now() < DEADLINE - 10.0:
        # Precompute small primes for trial division
        SIEVE_LIMIT = 100000
        is_prime_small = bytearray(b'\x01') * (SIEVE_LIMIT + 1)
        is_prime_small[0] = is_prime_small[1] = 0
        for i in range(2, int(math.isqrt(SIEVE_LIMIT)) + 1):
            if is_prime_small[i]:
                is_prime_small[i*i::i] = bytearray(len(is_prime_small[i*i::i]))
        small_primes = [i for i in range(2, SIEVE_LIMIT + 1) if is_prime_small[i]]
        
        def is_prime_mr(n):
            if n < 2:
                return False
            if n < SIEVE_LIMIT + 1:
                return bool(is_prime_small[n])
            for p in small_primes[:20]:
                if n % p == 0:
                    return n == p
            r, d = 0, n - 1
            while d % 2 == 0:
                r += 1
                d //= 2
            witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
            for a in witnesses:
                if a >= n:
                    continue
                x = pow(a, d, n)
                if x == 1 or x == n - 1:
                    continue
                for _ in range(r - 1):
                    x = (x * x) % n
                    if x == n - 1:
                        break
                else:
                    return False
            return True
        
        def pollard_rho(n):
            if n % 2 == 0: return 2
            if n % 3 == 0: return 3
            if n % 5 == 0: return 5
            while True:
                x = random.randint(2, n - 1)
                y = x
                c = random.randint(1, n - 1)
                d = 1
                while d == 1:
                    x = (x * x + c) % n
                    y = (y * y + c) % n
                    y = (y * y + c) % n
                    d = math.gcd(abs(x - y), n)
                if d != n:
                    return d
        
        def factorize(n):
            if n == 1: return []
            if is_prime_mr(n): return [n]
            d = pollard_rho(n)
            return factorize(d) + factorize(n // d)
        
        def phi_large(n):
            if n == 1: return 1
            temp = n
            result = n
            for p in small_primes:
                if p * p > temp: break
                if temp % p == 0:
                    while temp % p == 0:
                        temp //= p
                    result -= result // p
            if temp > 1:
                if is_prime_mr(temp):
                    result -= result // temp
                else:
                    factors = factorize(temp)
                    for p in set(factors):
                        result -= result // p
            return result
        
        def check_run(s, max_k=200):
            seen = set()
            k = 0
            while k < max_k:
                val = phi_large(s + k)
                if val in seen:
                    break
                seen.add(val)
                k += 1
            return k
        
        # Search in blocks beyond N
        # Strategy: check every 1000th number, and if we find a good run, check neighbors
        search_start = N + 1
        search_end = min(10**8, 10**12 - 200)
        
        for s in range(search_start, search_end + 1, 1000):
            if now() > DEADLINE - 5.0:
                break
            k = check_run(s)
            if k > best_k:
                best_k = k
                best_s = s
                # If we found a good run, check neighbors
                if k >= max(10, best_k - 2):
                    for delta in range(-10, 11):
                        if now() > DEADLINE - 5.0:
                            break
                        s2 = s + delta
                        if 1 <= s2 and s2 + 199 <= 10**12:
                            k2 = check_run(s2)
                            if k2 > best_k:
                                best_k = k2
                                best_s = s2
        
        # Also try some random starts in larger range
        if now() < DEADLINE - 3.0:
            random.seed(42)
            for _ in range(500):
                if now() > DEADLINE - 3.0:
                    break
                s = random.randint(10**8, 10**12 - 200)
                k = check_run(s)
                if k > best_k:
                    best_k = k
                    best_s = s
    
    return {'start': best_s, 'length': best_k}
