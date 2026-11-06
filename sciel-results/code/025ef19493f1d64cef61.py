import math
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    
    # Pre-compute small primes for trial division
    SMALL_PRIME_LIMIT = 50000
    small_primes = []
    is_sm = [True] * (SMALL_PRIME_LIMIT + 1)
    is_sm[0] = is_sm[1] = False
    for i in range(2, SMALL_PRIME_LIMIT + 1):
        if is_sm[i]:
            small_primes.append(i)
            if i * i <= SMALL_PRIME_LIMIT:
                for j in range(i * i, SMALL_PRIME_LIMIT + 1, i):
                    is_sm[j] = False
    
    # Build a fast phi function using precomputed smallest prime factor
    SPF_LIMIT = 2_000_000
    spf = list(range(SPF_LIMIT + 1))
    for i in range(2, int(SPF_LIMIT**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, SPF_LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def phi_fast(n):
        """Compute phi(n) using SPF for n <= SPF_LIMIT, fallback otherwise."""
        if n <= SPF_LIMIT:
            result = n
            while n > 1:
                p = spf[n]
                result -= result // p
                while n % p == 0:
                    n //= p
            return result
        else:
            return phi_slow(n)
    
    def phi_slow(n):
        """Fallback phi computation for large n."""
        if n == 1:
            return 1
        result = n
        temp = n
        for p in small_primes:
            if p * p > temp:
                break
            if temp % p == 0:
                result -= result // p
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            result -= result // temp
        return result
    
    def check_window(s, max_len=200, max_n=10**12):
        """Check how many consecutive integers starting from s have distinct phi values."""
        seen = set()
        k = 0
        for i in range(max_len):
            v = s + i
            if v > max_n:
                break
            pv = phi_fast(v) if v <= SPF_LIMIT else phi_slow(v)
            if pv in seen:
                break
            seen.add(pv)
            k += 1
        return k
    
    def find_best_in_range(start, end, step=1):
        """Find the best starting point in [start, end] by sampling."""
        best_k = 0
        best_s = start
        count = 0
        for s in range(start, end + 1, step):
            if time.time() - start_time > DEADLINE:
                break
            k = check_window(s, 200)
            if k > best_k:
                best_k = k
                best_s = s
            count += 1
        return best_s, best_k
    
    best_k = 1
    best_s = 1
    max_n = 10**12
    
    # Strategy 1: Systematic scan of small ranges where phi is fast (SPF available)
    # Scan in chunks to find promising regions
    chunk_size = 10000
    scan_limit = min(SPF_LIMIT, max_n)
    
    for base in range(1, scan_limit + 1, chunk_size):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Sample multiple points in this chunk
        samples = []
        for offset in [0, chunk_size//4, chunk_size//2, 3*chunk_size//4, chunk_size-1]:
            s = base + offset
            if 1 <= s <= max_n - 200:
                samples.append(s)
        
        # Check each sample and track best
        for s in samples:
            if time.time() - start_time > DEADLINE - 10:
                break
            k = check_window(s, 200)
            if k > best_k:
                best_k = k
                best_s = s
    
    # Strategy 2: Local optimization around best found so far
    if best_k > 1:
        # Search left and right from best_s
        for direction in [-1, 1]:
            for dist in range(1, 500):
                if time.time() - start_time > DEADLINE - 5:
                    break
                ns = best_s + direction * dist
                if ns < 1 or ns > max_n - 200:
                    break
                k = check_window(ns, 200)
                if k > best_k:
                    best_k = k
                    best_s = ns
                    break  # Restart search from new best
    
    # Strategy 3: Structured search near primorials and smooth numbers
    primorial_values = [1]
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]:
        if primorial_values[-1] * p > max_n:
            break
        primorial_values.append(primorial_values[-1] * p)
    
    for pm in primorial_values:
        if time.time() - start_time > DEADLINE - 5:
            break
        # Check around multiples of primorial
        for mult in [1, 2, 3, 5, 7, 10, 13, 17, 20, 25, 30]:
            s = pm * mult
            if s > max_n - 200 or s < 1:
                continue
            k = check_window(s, 200)
            if k > best_k:
                best_k = k
                best_s = s
            # Also try nearby offsets
            for delta in range(-10, 11):
                if time.time() - start_time > DEADLINE - 5:
                    break
                ns = s + delta
                if 1 <= ns <= max_n - 200:
                    nk = check_window(ns, 200)
                    if nk > best_k:
                        best_k = nk
                        best_s = ns
    
    # Strategy 4: Random search with better distribution
    random.seed(42)
    
    # Use log-uniform distribution to cover more range
    for iteration in range(3000):
        if time.time() - start_time > DEADLINE - 2:
            break
        
        # Log-uniform sampling
        log_s = random.uniform(0, 12)  # log10(1) to log10(10^12)
        s = int(10 ** log_s)
        s = max(1, min(s, max_n - 200))
        
        # Add some jitter
        jitter = random.randint(-100, 100)
        s = max(1, min(s + jitter, max_n - 200))
        
        k = check_window(s, 200)
        if k > best_k:
            best_k = k
            best_s = s
            # Local optimization around new best
            for direction in [-1, 1]:
                for dist in range(1, 100):
                    if time.time() - start_time > DEADLINE - 2:
                        break
                    ns = best_s + direction * dist
                    if ns < 1 or ns > max_n - 200:
                        break
                    nk = check_window(ns, 200)
                    if nk > best_k:
                        best_k = nk
                        best_s = ns
                        break
    
    # Final verification: ensure the returned window is valid
    # Re-check the best window to be safe
    final_k = check_window(best_s, 200)
    if final_k > best_k:
        best_k = final_k
    else:
        # If our stored best_k was optimistic, use the verified value
        best_k = min(best_k, final_k)
    
    # Ensure we don't exceed bounds
    if best_s + best_k - 1 > max_n:
        best_k = max_n - best_s + 1
    
    return {'start': best_s, 'length': best_k}
