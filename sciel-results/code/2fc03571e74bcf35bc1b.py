import time
import math
import random
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    # The evaluator kills at 150s. We need a safety margin.
    # Let's aim to finish computation by 145s to be safe.
    DEADLINE = 145.0

    def now():
        return time.time()

    # --- Precompute small primes for trial division (for large n) ---
    LIMIT_TRIAL = 10**6
    is_prime_small = bytearray(b'\x01') * (LIMIT_TRIAL + 1)
    is_prime_small[0] = is_prime_small[1] = 0
    for i in range(2, int(math.isqrt(LIMIT_TRIAL)) + 1):
        if is_prime_small[i]:
            is_prime_small[i*i::i] = bytearray(len(is_prime_small[i*i::i]))
    small_primes = [i for i in range(2, LIMIT_TRIAL + 1) if is_prime_small[i]]

    def is_prime_mr(n):
        if n < 2:
            return False
        if n < LIMIT_TRIAL:
            return bool(is_prime_small[n])
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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
        if n == 1:
            return []
        if is_prime_mr(n):
            return [n]
        d = pollard_rho(n)
        return factorize(d) + factorize(n // d)

    def phi_large(n):
        if n == 1:
            return 1
        temp = n
        result = n
        for p in small_primes:
            if p * p > temp:
                break
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

    def phi_batch(start, count):
        """Compute phi for count consecutive integers starting at start."""
        return [phi_large(start + i) for i in range(count)]

    def max_distinct_run(vals, max_k=200):
        """Find the longest run of consecutive distinct values in vals."""
        n = len(vals)
        if n == 0:
            return 0, 0
        best_k = 0
        best_idx = 0
        counts = {}
        left = 0
        
        for right in range(n):
            v = vals[right]
            counts[v] = counts.get(v, 0) + 1
            
            # Shrink window until all distinct
            while counts[v] > 1:
                v_left = vals[left]
                counts[v_left] -= 1
                if counts[v_left] == 0:
                    del counts[v_left]
                left += 1
            
            curr_k = right - left + 1
            if curr_k > best_k:
                best_k = curr_k
                best_idx = left
                if best_k >= max_k:
                    break
        return best_k, best_idx

    best_k = 1
    best_s = 1

    # --- Phase 1: Fast NumPy Sieve with Sliding Window ---
    # We want to scan a large range. 
    # The current program uses N=30*10^6. 
    # Inspiration 1 uses N=10^7.
    # Let's try to be a bit more aggressive if time permits, but 30M is a good balance.
    # The bottleneck is the sieve and the phi calculation loop.
    
    N = 80 * 10**6  # 80 million

    try:
        # Initialize phi array
        phi_arr = np.arange(N + 1, dtype=np.int64)

        # Find primes up to N using a fast bytearray sieve
        is_prime = bytearray(b'\x01') * (N + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(N)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))

        # Get primes as a numpy array for efficient iteration
        primes_np = np.frombuffer(is_prime, dtype=np.uint8).nonzero()[0]
        primes_np = primes_np[primes_np >= 2]

        # Update phi for each prime using vectorized operations with slicing
        # This is faster than creating index arrays
        for p in primes_np:
            p = int(p)
            phi_arr[p::p] -= phi_arr[p::p] // p

        # Find max distinct run using sliding window
        vals = phi_arr[1:]
        k, idx = max_distinct_run(vals, 200)
        if k > best_k:
            best_k = k
            best_s = idx + 1

    except Exception as e:
        # Fallback to smaller N
        N = 30 * 10**6
        try:
            phi_arr = np.arange(N + 1, dtype=np.int64)
            is_prime = bytearray(b'\x01') * (N + 1)
            is_prime[0] = is_prime[1] = 0
            for i in range(2, int(math.isqrt(N)) + 1):
                if is_prime[i]:
                    is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
            primes_np = np.frombuffer(is_prime, dtype=np.uint8).nonzero()[0]
            primes_np = primes_np[primes_np >= 2]
            for p in primes_np:
                p = int(p)
                phi_arr[p::p] -= phi_arr[p::p] // p
            vals = phi_arr[1:]
            k, idx = max_distinct_run(vals, 200)
            if k > best_k:
                best_k = k
                best_s = idx + 1
        except:
            pass

    # --- Phase 2: Targeted Search in Larger Ranges ---
    # Scan contiguous blocks in strategically chosen regions to find longer runs.
    # Each block is 2000 wide, giving 10x more coverage than single-point checks.
    if now() < DEADLINE - 10.0:
        random.seed(42)
        
        block_size = 2000
        regions = []
        
        # Powers of 10 and nearby
        for exp in range(6, 13):
            base = 10 ** exp
            for offset in [-5000, -2500, -1000, -500, 0, 500, 1000, 2500, 5000]:
                s = base + offset
                if 1 <= s <= 10**12 - block_size:
                    regions.append(s)
        
        # Random large ranges - use contiguous blocks
        for _ in range(30):
            s = random.randint(10**6, 10**12 - block_size)
            regions.append(s)
        
        # Medium ranges
        for _ in range(20):
            s = random.randint(10**5, 10**8)
            regions.append(s)
        
        # Remove duplicates and sort
        regions = sorted(set(regions), reverse=True)
        
        for s in regions:
            if now() > DEADLINE - 5.0:
                break
            if s < 1 or s + block_size > 10**12:
                continue
            
            try:
                vals = phi_batch(s, block_size)
            except Exception:
                continue
            
            k, idx = max_distinct_run(vals, 200)
            if k > best_k:
                best_k = k
                best_s = s + idx
                if best_k == 200:
                    break
    
    # --- Phase 3: Fine-tune around best found ---
    # Scan a wider neighborhood with overlapping blocks to catch runs that
    # span across the initial best location.
    if now() < DEADLINE - 3.0 and best_k < 200:
        window = 10000
        s_min = max(1, best_s - window)
        s_max = min(10**12 - 200, best_s + window)
        
        # Use overlapping blocks of size 2000 with stride 500
        block_size = 2000
        stride = 500
        n_samples = (s_max - s_min + 1) // stride + 1
        n_samples = min(n_samples, 100)
        
        for i in range(n_samples):
            if now() > DEADLINE - 2.0:
                break
            s = s_min + i * stride
            s = int(s)
            if s < 1 or s + block_size > 10**12:
                continue
            try:
                vals = phi_batch(s, block_size)
            except Exception:
                continue
            k, idx = max_distinct_run(vals, 200)
            if k > best_k:
                best_k = k
                best_s = s + idx
                if best_k == 200:
                    break

    return {'start': best_s, 'length': best_k}
