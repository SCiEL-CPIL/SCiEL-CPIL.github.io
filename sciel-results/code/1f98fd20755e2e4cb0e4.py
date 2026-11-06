"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import sys

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
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
    SMALL_PRIME_LIMIT = 100000
    small_primes = sieve_primes(SMALL_PRIME_LIMIT)
    
    def is_prime(n):
        """Deterministic Miller-Rabin for n < 3.3e24."""
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        # Miller-Rabin with deterministic witnesses for 64-bit
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        # Witnesses sufficient for n < 3.3e24
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
    
    def phi_sieve(limit):
        """Compute phi for all numbers 1..limit using sieve method.
        Returns list where phi_arr[i] = phi(i)."""
        phi_arr = list(range(limit + 1))
        for i in range(2, limit + 1):
            if phi_arr[i] == i:  # i is prime
                for j in range(i, limit + 1, i):
                    phi_arr[j] -= phi_arr[j] // i
        return phi_arr
    
    def check_run_with_phi(start, max_length, max_n, phi_arr=None, phi_arr_start=0):
        """Check consecutive integers from start for distinct phi values.
        If phi_arr is provided, use it for fast lookup when applicable.
        Returns (length, start) of longest run found."""
        seen = set()
        k = 0
        for i in range(max_length):
            n = start + i
            if n > max_n:
                break
            if phi_arr is not None and phi_arr_start <= n <= phi_arr_start + len(phi_arr) - 1:
                v = phi_arr[n - phi_arr_start]
            else:
                v = phi(n)
            if v in seen:
                break
            seen.add(v)
            k += 1
        return k, start
    
    def find_longest_run_in_range(start, end, max_length, max_n, phi_arr=None, phi_arr_start=0):
        """Find the longest run of distinct phi values in [start, end].
        Returns (best_k, best_s) for runs starting in this range."""
        best_k = 1
        best_s = start
        s = start
        while s <= end:
            if time.time() > deadline - 5:
                break
            k, _ = check_run_with_phi(s, max_length, max_n, phi_arr, phi_arr_start)
            if k > best_k:
                best_k = k
                best_s = s
            # Jump ahead by k (since positions s+1..s+k-1 can't start a longer run
            # that beats k, but they might start a different run of length k)
            # Actually, we should jump by max(1, k) to be safe, but let's be more aggressive
            # If we found a run of length k starting at s, then s+1 can't have a run
            # longer than k (it would need k+1 distinct values in a window that overlaps)
            # Actually that's not quite right. Let's just jump by 1 for correctness
            # but we can optimize: if k >= max_length, we're done
            if k >= max_length:
                break
            s += 1
        return best_k, best_s
    
    best_k = 1
    best_s = 1
    
    # Phase 1: Sieve-based dense search in small ranges
    # Use phi_sieve for fast computation up to 500,000
    sieve_limit = 500000
    phi_arr_small = phi_sieve(sieve_limit)
    
    # Dense search using precomputed phi values with smarter jumping
    s = 1
    while s <= sieve_limit:
        if time.time() > deadline - 5:
            break
        k, _ = check_run_with_phi(s, 200, MAX_N, phi_arr_small, 1)
        if k > best_k:
            best_k = k
            best_s = s
        # If we found a run of length k, we can skip ahead by k-1
        # because positions s+1 through s+k-1 cannot start a longer run
        # that extends beyond s+k-1 (they would need distinct values in a window
        # that overlaps with the already-checked window)
        # But to be safe, let's jump by max(1, k//2) to balance speed and coverage
        if k >= 200:
            break
        s += max(1, k // 2)
    
    # Continue dense search with individual phi calls for the rest of the small range
    if time.time() < deadline - 10:
        # Extend search to 2,000,000 using individual phi computations
        # but only check every 10th number to save time
        for s in range(sieve_limit + 1, 2000001, 10):
            if time.time() > deadline - 5:
                break
            k, _ = check_run_with_phi(s, 200, MAX_N)
            if k > best_k:
                best_k = k
                best_s = s
    
    # Phase 2: Targeted searches around structured numbers
    # Numbers that are products of small primes, primorials, and nearby numbers
    if time.time() < deadline - 10:
        # Generate some primorials and products of small primes
        primorials = [2]
        prod = 2
        for p in small_primes[1:]:  # skip 2, start from 3
            prod *= p
            if prod > 10**12:
                break
            primorials.append(prod)
        
        # Also include products of first n primes (not just primorials)
        structured_starts = set()
        
        # Add primorials with wider offsets
        for prim in primorials:
            for offset in range(-200, 201, 5):
                s = prim + offset
                if 1 <= s <= MAX_N - 200:
                    structured_starts.add(s)
        
        # Powers of 2
        for exp in range(1, 40):
            p2 = 2 ** exp
            if p2 > MAX_N:
                break
            for offset in range(-200, 201, 5):
                s = p2 + offset
                if 1 <= s <= MAX_N - 200:
                    structured_starts.add(s)
        
        # Powers of 3
        for exp in range(1, 26):
            p3 = 3 ** exp
            if p3 > MAX_N:
                break
            for offset in range(-200, 201, 5):
                s = p3 + offset
                if 1 <= s <= MAX_N - 200:
                    structured_starts.add(s)
        
        # Products of first few primes (not necessarily consecutive)
        # e.g., 2*3*5, 2*3*7, 2*5*7, 3*5*7, etc.
        for i in range(len(small_primes)):
            if i > 15:
                break
            for j in range(i+1, min(i+10, len(small_primes))):
                prod_ij = small_primes[i] * small_primes[j]
                if prod_ij > MAX_N:
                    break
                for k_idx in range(j+1, min(j+10, len(small_primes))):
                    prod_ijk = prod_ij * small_primes[k_idx]
                    if prod_ijk > MAX_N:
                        break
                    for offset in range(-100, 101, 5):
                        s = prod_ijk + offset
                        if 1 <= s <= MAX_N - 200:
                            structured_starts.add(s)
        
        # Check structured starts (limit to 5000 most promising)
        structured_starts = list(structured_starts)
        structured_starts.sort()
        for s in structured_starts[:5000]:
            if time.time() > deadline - 5:
                break
            k, _ = check_run_with_phi(s, 200, MAX_N)
            if k > best_k:
                best_k = k
                best_s = s
    
    # Phase 3: Random sampling in larger ranges with adaptive strategy
    if time.time() < deadline - 10:
        # Sample random starting points in various ranges
        # Focus more on ranges where we're more likely to find longer runs
        ranges = [
            (2 * 10**5, 10**6),
            (10**6, 10**7),
            (10**7, 10**8),
            (10**8, 10**9),
            (10**9, 10**10),
            (10**10, 10**11),
            (10**11, 10**12),
        ]
        
        # Allocate samples based on remaining time
        total_samples = 1500
        samples_per_range = total_samples // len(ranges)
        
        for lo, hi in ranges:
            if time.time() > deadline - 5:
                break
            for _ in range(samples_per_range):
                if time.time() > deadline - 5:
                    break
                s = random.randint(lo, min(hi, int(MAX_N) - 200))
                k, _ = check_run_with_phi(s, 200, MAX_N)
                if k > best_k:
                    best_k = k
                    best_s = s
    
    # Phase 4: Local refinement around the best found
    if best_k > 3 and time.time() < deadline - 10:
        # Search around the best found start with a wider window
        # Use adaptive window size based on how much time we have
        window = 10000
        # Check every 2nd number to save time, then refine if needed
        for offset in range(-window, window + 1, 2):
            if time.time() > deadline - 5:
                break
            s = best_s + offset
            if s < 1 or s + 199 > MAX_N:
                continue
            k, _ = check_run_with_phi(s, 200, MAX_N)
            if k > best_k:
                best_k = k
                best_s = s
        # If we found a better one, do a finer search around it
        if time.time() < deadline - 5:
            for offset in range(-100, 101, 1):
                if time.time() > deadline - 5:
                    break
                s = best_s + offset
                if s < 1 or s + 199 > MAX_N:
                    continue
                k, _ = check_run_with_phi(s, 200, MAX_N)
                if k > best_k:
                    best_k = k
                    best_s = s
    
    # Phase 5: If we have time, do a second round of random sampling
    # and also try to extend the current best run
    if time.time() < deadline - 5:
        # More random samples, focusing on ranges where we might find longer runs
        remaining_time = deadline - 5 - time.time()
        num_extra_samples = int(remaining_time * 20)  # ~20 samples per second
        for _ in range(num_extra_samples):
            if time.time() > deadline - 5:
                break
            # Bias towards larger numbers where runs might be longer
            log_lo = 5
            log_hi = 12
            log_s = random.uniform(log_lo, log_hi)
            s = int(10 ** log_s)
            s = max(1, min(s, int(MAX_N) - 200))
            k, _ = check_run_with_phi(s, 200, MAX_N)
            if k > best_k:
                best_k = k
                best_s = s
        
        # Try to extend the current best run by checking nearby starts
        if time.time() < deadline - 3:
            # Check if we can start earlier or later to get a longer run
            for delta in range(-50, 51):
                if time.time() > deadline - 3:
                    break
                s = best_s + delta
                if s < 1 or s + 199 > MAX_N:
                    continue
                k, _ = check_run_with_phi(s, 200, MAX_N)
                if k > best_k:
                    best_k = k
                    best_s = s
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
