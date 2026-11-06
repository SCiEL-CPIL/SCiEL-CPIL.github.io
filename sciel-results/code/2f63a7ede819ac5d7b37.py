"""Rewritten program: search for long runs of consecutive integers with distinct totients."""
# EVOLVE-BLOCK-START
import time
import math
import random
import sys

def construct():
    deadline = time.time() + 105  # Safety margin before 120s limit
    random.seed(42)
    
    # Miller-Rabin primality test
    def is_prime(n):
        if n < 2:
            return False
        # Small primes check
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        
        # Write n-1 as d * 2^s
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        
        # Bases sufficient for n < 2^64
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

    # Pollard's Rho algorithm
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
        """Compute Euler's totient using Pollard's Rho."""
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
    
    def compute_phi_range(start, length):
        """Compute phi values for range [start, start+length-1]."""
        return [phi_fast(start + i) for i in range(length)]
    
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
    
    best_overall = (1, 1)  # (length, start)
    
    def check_and_update(start, length):
        nonlocal best_overall
        if length < best_overall[0]:
            return
        vals = compute_phi_range(start, length)
        k, offset = find_best_run(vals, length)
        if k > best_overall[0]:
            best_overall = (k, start + offset)
    
    # Strategy 1: Small range exhaustive scan (1 to 100000) - faster now
    if time.time() < deadline:
        scan_end = 100000
        vals = compute_phi_range(1, scan_end)
        best_k, best_s = find_best_run(vals, 200)
        if best_k > best_overall[0]:
            best_overall = (best_k, best_s + 1)
    
    # Strategy 2: Targeted search around "smooth" numbers and primorials
    # Long runs are more likely when numbers have varied factorizations
    # Try starts near products of small primes, near powers of 2, etc.
    if time.time() < deadline:
        candidates = []
        # Near primorials (precompute small primes for this part)
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        primorial = 1
        for p in small_primes:
            primorial *= p
            if primorial > 10**12:
                break
            for delta in range(-50, 51, 5):
                s = primorial + delta
                if 1 <= s <= 10**12 - 200:
                    candidates.append(s)
        
        # Near powers of 2
        for e in range(1, 40):
            p2 = 1 << e
            for delta in range(-50, 51, 5):
                s = p2 + delta
                if 1 <= s <= 10**12 - 200:
                    candidates.append(s)
        
        # Near powers of 3
        for e in range(1, 26):
            p3 = 3 ** e
            for delta in range(-50, 51, 5):
                s = p3 + delta
                if 1 <= s <= 10**12 - 200:
                    candidates.append(s)
        
        # Deduplicate and sort
        candidates = sorted(set(candidates))
        
        for s in candidates:
            if time.time() >= deadline:
                break
            if s < 1 or s + 200 > 10**12:
                continue
            check_and_update(s, 200)
    
    # Strategy 3: Random sampling in large ranges - much faster now
    if time.time() < deadline:
        attempts = 0
        while time.time() < deadline - 5 and attempts < 500000:
            attempts += 1
            # Sample from various ranges
            r = random.random()
            if r < 0.3:
                # Mid-range
                s = random.randint(10**6, 10**9)
            elif r < 0.6:
                # Upper range
                s = random.randint(10**9, 10**12 - 200)
            elif r < 0.8:
                # Lower-mid range
                s = random.randint(10**4, 10**7)
            else:
                # Very high range
                s = random.randint(10**11, 10**12 - 200)
            
            if s < 1 or s + 200 > 10**12:
                continue
            check_and_update(s, 200)
    
    # Strategy 4: Local search - if we found a good run, explore nearby
    if time.time() < deadline and best_overall[0] > 5:
        center = best_overall[1]
        # Denser local search
        for delta in range(-500, 501, 2):
            if time.time() >= deadline:
                break
            s = center + delta
            if 1 <= s <= 10**12 - 200:
                check_and_update(s, 200)
    
    # Strategy 5: If we have time, do a denser scan in promising regions
    if time.time() < deadline - 10:
        # Focus on region around current best
        if best_overall[0] >= 3:
            lo = max(1, best_overall[1] - 10000)
            hi = min(10**12 - 200, best_overall[1] + 10000)
            # Scan in chunks
            chunk = 2000
            s = lo
            while s + chunk <= hi + 1 and time.time() < deadline - 5:
                vals = compute_phi_range(s, chunk)
                k, offset = find_best_run(vals, 200)
                if k > best_overall[0]:
                    best_overall = (k, s + offset)
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
