import time
import math
import random
import sys

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    random.seed(42)
    
    # Precompute small primes up to 1e6 for fast factorization
    def sieve_primes(limit):
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i * i, limit + 1, i):
                    is_prime[j] = 0
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    primes = sieve_primes(1000000)
    
    def phi_fast(x):
        """Compute Euler's totient using precomputed primes."""
        if x <= 0:
            return 0
        result = x
        y = x
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result

    def check_run(start):
        """Check phi values for start to start+199, return length of distinct run."""
        seen = set()
        k = 0
        for i in range(200):
            val = phi_fast(start + i)
            if val in seen:
                break
            seen.add(val)
            k += 1
        return k

    def batch_phi(L, R):
        """Compute phi for all n in [L, R] and return list of values."""
        vals = []
        for n in range(L, R + 1):
            vals.append(phi_fast(n))
        return vals

    def find_best_run_in_vals(vals, offset):
        """
        Find the longest run of consecutive distinct values in vals.
        Returns (length, start_index_in_vals).
        """
        n = len(vals)
        if n == 0:
            return 0, 0
        best_k = 1
        best_s = 0
        seen = set()
        current_k = 1
        seen.add(vals[0])
        
        for i in range(1, n):
            if vals[i] in seen:
                # Reset run
                if current_k > best_k:
                    best_k = current_k
                    best_s = i - current_k
                # Start new run
                seen = {vals[i]}
                current_k = 1
            else:
                seen.add(vals[i])
                current_k += 1
                # Update best if current run is longer and we have enough room
                # Note: We want the run to be within the array.
                if current_k > best_k:
                    best_k = current_k
                    best_s = i - current_k + 1
        
        # Check the last run
        if current_k > best_k:
            best_k = current_k
            best_s = n - current_k
            
        return best_k, best_s

    best_k = 1
    best_s = 1
    
    def update_best(start, k):
        nonlocal best_k, best_s
        if k > best_k:
            best_k = k
            best_s = start

    # Strategy 1: Small range exhaustive scan (1 to 10000)
    # This is fast enough with batch_phi
    if time.time() < deadline:
        L, R = 1, 10000
        vals = batch_phi(L, R)
        k, idx = find_best_run_in_vals(vals, L)
        update_best(L + idx, k)

    # Strategy 2: Random sampling with local refinement
    attempts = 0
    while time.time() < deadline - 10 and attempts < 20000:
        attempts += 1
        # Sample from various ranges
        r = random.random()
        if r < 0.2:
            s = random.randint(10**4, 10**7)
        elif r < 0.4:
            s = random.randint(10**7, 10**9)
        elif r < 0.6:
            s = random.randint(10**9, 10**11)
        elif r < 0.8:
            s = random.randint(10**11, 10**12 - 200)
        else:
            s = random.randint(10**5, 10**12 - 200)
            
        if s < 1 or s + 200 > 10**12:
            continue
        
        # Quick check
        k = check_run(s)
        
        # If promising, do a local dense scan
        if k > 5:
            # Scan a window around s
            # Window size 2000 is ~4000 phi calculations, which is fast
            win_size = 1000
            L = max(1, s - win_size)
            R = min(10**12, s + win_size + 200)
            
            if R - L < 100:
                continue
                
            vals = batch_phi(L, R)
            k_local, idx_local = find_best_run_in_vals(vals, L)
            start_local = L + idx_local
            
            # Ensure the run is fully within bounds
            if start_local + k_local - 1 <= 10**12:
                update_best(start_local, k_local)
            else:
                # Truncate if necessary
                valid_k = 10**12 - start_local + 1
                if valid_k > 0:
                    update_best(start_local, valid_k)

    # Ensure validity
    s = best_s
    k = best_k
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
