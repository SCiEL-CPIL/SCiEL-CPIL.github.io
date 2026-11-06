"""Rewritten program: search for long runs of consecutive integers with distinct totients."""
# EVOLVE-BLOCK-START
import time
import math
import random
import sys

def construct():
    deadline = time.time() + 110  # Safety margin before 150s limit
    random.seed(42)
    
    # Miller-Rabin primality test
    def is_prime(n):
        if n < 2:
            return False
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in bases:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
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
        factors = []
        def _factor(m, factors):
            if m == 1:
                return
            if is_prime(m):
                factors.append(m)
                return
            d = pollard_rho(m)
            _factor(d, factors)
            _factor(m // d, factors)
        _factor(n, factors)
        return factors

    def phi_fast(x):
        if x <= 0:
            return 0
        if x == 1:
            return 1
        factors = factorize(x)
        result = x
        seen = set()
        for p in factors:
            if p not in seen:
                seen.add(p)
                result -= result // p
        return result

    def find_best_run(values, max_len=200):
        """Find the longest run of consecutive distinct phi values in values."""
        n = len(values)
        best_k = 1
        best_s = 0
        for s in range(n - max_len + 1):
            seen = set()
            k = 0
            for i in range(s, min(s + max_len, n)):
                if values[i] in seen:
                    break
                seen.add(values[i])
                k += 1
            if k > best_k:
                best_k = k
                best_s = s
        return best_k, best_s

    def find_best_run_fast(values, max_len=200):
        """Faster: find longest run where all phi values are distinct."""
        n = len(values)
        best_k = 1
        best_s = 0
        # Use a sliding window approach with a dict tracking last seen
        last_seen = {}
        left = 0
        for right in range(n):
            v = values[right]
            if v in last_seen and last_seen[v] >= left:
                # Shrink window
                left = last_seen[v] + 1
            last_seen[v] = right
            current_len = right - left + 1
            if current_len > best_k:
                best_k = current_len
                best_s = left
            if best_k >= max_len:
                break
        return min(best_k, max_len), best_s

    def compute_phi_range_sieve(lo, hi):
        """Compute phi(n) for all n in [lo, hi] using segmented sieve."""
        length = hi - lo + 1
        # phi[n - lo] will store phi(n)
        phi_arr = list(range(lo, hi + 1))  # phi[n] starts as n
        # For each prime p, for each multiple of p in [lo,hi], multiply phi by (1 - 1/p)
        limit = int(math.isqrt(hi)) + 1
        # Sieve primes up to limit
        is_comp = [False] * (limit + 1)
        primes = []
        for i in range(2, limit + 1):
            if not is_comp[i]:
                primes.append(i)
                if i * i <= limit:
                    for j in range(i * i, limit + 1, i):
                        is_comp[j] = True
        
        for p in primes:
            # First multiple of p >= lo
            first = ((lo + p - 1) // p) * p
            for m in range(first, hi + 1, p):
                idx = m - lo
                if phi_arr[idx] % p == 0:
                    phi_arr[idx] = phi_arr[idx] // p * (p - 1)
        
        # Handle numbers that are prime and > limit
        # For any phi_arr[idx] that still equals the original number, it's prime
        for i in range(length):
            n = lo + i
            if n >= 2 and phi_arr[i] == n:
                # n is prime
                phi_arr[i] = n - 1
        
        return phi_arr

    best_overall = (1, 1)  # (length, start)

    def check_and_update(start, length):
        nonlocal best_overall
        if length < best_overall[0]:
            return
        vals = compute_phi_range_sieve(start, start + length - 1)
        k, offset = find_best_run_fast(vals, length)
        if k > best_overall[0]:
            best_overall = (k, start + offset)

    # Strategy 1: Sieve scan of a large contiguous range
    # Start with a moderate range to get a baseline
    if time.time() < deadline:
        scan_size = 200000
        lo = 1
        hi = lo + scan_size - 1
        vals = compute_phi_range_sieve(lo, hi)
        best_k, best_s = find_best_run_fast(vals, 200)
        if best_k > best_overall[0]:
            best_overall = (best_k, lo + best_s)
    
    # Strategy 2: Sieve scan in multiple windows across different ranges
    if time.time() < deadline:
        window_size = 50000
        # Scan several windows in different regions
        regions = [
            (1, 500000),
            (10**6, 10**6 + 500000),
            (10**7, 10**7 + 500000),
            (10**8, 10**8 + 500000),
            (10**9, 10**9 + 500000),
            (10**10, 10**10 + 500000),
            (10**11, 10**11 + 500000),
            (10**12 - 500000, 10**12),
        ]
        for reg_lo, reg_hi in regions:
            if time.time() >= deadline - 5:
                break
            s = reg_lo
            while s + window_size <= reg_hi + 1 and time.time() < deadline - 5:
                check_and_update(s, window_size)
                s += window_size
    
    # Strategy 3: Targeted search around interesting numbers
    if time.time() < deadline:
        candidates = []
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        primorial = 1
        for p in small_primes:
            primorial *= p
            if primorial > 10**12:
                break
            for delta in range(-100, 101, 10):
                s = primorial + delta
                if 1 <= s <= 10**12 - 200:
                    candidates.append(s)
        
        for e in range(1, 40):
            p2 = 1 << e
            for delta in range(-100, 101, 10):
                s = p2 + delta
                if 1 <= s <= 10**12 - 200:
                    candidates.append(s)
        
        for e in range(1, 26):
            p3 = 3 ** e
            for delta in range(-100, 101, 10):
                s = p3 + delta
                if 1 <= s <= 10**12 - 200:
                    candidates.append(s)
        
        candidates = sorted(set(candidates))
        
        for s in candidates:
            if time.time() >= deadline - 5:
                break
            if s < 1 or s + 200 > 10**12:
                continue
            check_and_update(s, 200)
    
    # Strategy 4: Random sampling with sieve
    if time.time() < deadline:
        attempts = 0
        while time.time() < deadline - 5 and attempts < 10000:
            attempts += 1
            r = random.random()
            if r < 0.3:
                s = random.randint(10**6, 10**9)
            elif r < 0.6:
                s = random.randint(10**9, 10**12 - 500)
            elif r < 0.8:
                s = random.randint(10**4, 10**7)
            else:
                s = random.randint(10**11, 10**12 - 500)
            
            if s < 1 or s + 500 > 10**12:
                continue
            check_and_update(s, 500)
    
    # Strategy 5: Local search around best found
    if time.time() < deadline and best_overall[0] > 5:
        center = best_overall[1]
        for delta in range(-2000, 2001, 100):
            if time.time() >= deadline - 5:
                break
            s = center + delta
            if 1 <= s <= 10**12 - 500:
                check_and_update(s, 500)
    
    # Strategy 6: Denser scan around current best if time remains
    if time.time() < deadline - 10 and best_overall[0] >= 3:
        lo = max(1, best_overall[1] - 50000)
        hi = min(10**12 - 500, best_overall[1] + 50000)
        chunk = 10000
        s = lo
        while s + chunk <= hi + 1 and time.time() < deadline - 5:
            check_and_update(s, chunk)
            s += chunk
    
    # Ensure validity
    s = best_overall[1]
    k = best_overall[0]
    if s < 1:
        s = 1
    if s + k - 1 > 10**12:
        k = 10**12 - s + 1
        if k < 1:
            s = 1
            k = 1
    if k > 200:
        k = 200
    if k < 1:
        k = 1
    
    return {'start': s, 'length': k}

# EVOLVE-BLOCK-END
