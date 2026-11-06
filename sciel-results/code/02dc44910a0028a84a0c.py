import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    def now():
        return time.time()

    # --- Fast Factorization for Large N (Random Search Phase) ---
    
    # Precompute small primes for trial division (up to 10^5 is enough for quick checks)
    LIMIT_TRIAL = 10**5
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
        for p in small_primes[:10]: # Check first few small primes
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
        # Trial division with small primes
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
                # Factorize the remaining part
                factors = factorize(temp)
                for p in set(factors):
                    result -= result // p
        return result

    # --- Phase 1: NumPy Sieve for Small/Medium Range ---
    
    # Determine N based on time. 10^7 is safe.
    N = 10**7
    # Check if we have time/memory. 
    # np.arange(10^7) is ~80MB. OK.
    
    try:
        phi_arr = np.arange(N + 1, dtype=np.int64)
        # Sieve
        # For each i, if phi[i] == i, it's prime.
        # Optimization: Only iterate up to sqrt(N) for the outer loop?
        # No, we need to process all primes.
        # But we can stop outer loop at sqrt(N) and handle remaining?
        # Standard totient sieve:
        # for i in range(2, N+1):
        #   if phi[i] == i:
        #      for j in range(i, N+1, i):
        #          phi[j] -= phi[j] // i
        
        # Vectorized version:
        # We iterate i from 2 to N.
        # This is slow in Python if we do it 10^7 times.
        # Better: Get primes first?
        # Or just use the property that we only update if i is prime.
        
        # Let's use a standard loop but optimized.
        # Actually, 10^7 iterations of a simple check is ~1s.
        # The inner vectorized update is fast.
        
        for i in range(2, N + 1):
            if phi_arr[i] == i:
                # i is prime
                # Update multiples
                # indices = np.arange(i, N+1, i)
                # phi_arr[indices] -= phi_arr[indices] // i
                # This creates a temporary array.
                # Better:
                phi_arr[i::i] -= phi_arr[i::i] // i
            
            if now() > DEADLINE - 10.0:
                # If taking too long, reduce N? 
                # We are already in the loop. 
                # We can break and use partial? No, phi is wrong if incomplete.
                # So we must ensure N is small enough.
                # 10^7 should finish in < 5s.
                pass

        # Find max distinct run in phi_arr[1:]
        best_k = 1
        best_s = 1
        
        # Sliding window / Iteration
        # We can't use a set of size 10^7 easily if we want to be fast?
        # Actually, we just need the max run.
        # Iterate s from 1 to N.
        # For each s, expand k.
        # This is O(N * K_avg).
        
        # Optimization: 
        # If we find k, we can skip s+1...s+k-1? No, they might be part of a longer run starting earlier?
        # No, if run at s is length k, then run at s+1 is at most k-1 (if phi[s+k] != phi[s]).
        # Actually, distinctness is local.
        
        # Simple loop:
        seen = set()
        curr_k = 0
        for i in range(1, N + 1):
            v = phi_arr[i]
            if v in seen:
                # Reset? No, we need to find the max run ending at i?
                # Or starting at i?
                # The task asks for start s.
                # If we iterate s from 1 to N, and for each s find max k.
                # That is O(N * K).
                
                # Better: Single pass?
                # No, "distinct in window" is not a standard single-pass sliding window
                # because removing an element from the front might allow a new element
                # that was previously blocked?
                # No, if phi[s] == phi[s+k], then any window containing both is invalid.
                # So if we have a collision at distance d, any window of length > d containing both is invalid.
                
                # Let's stick to the simple O(N * K) but break early.
                pass

        # Actually, the previous code did:
        # for i in range(n):
        #   seen = set()
        #   for j in range(i, ...):
        #       if vals[j] in seen: break
        #       seen.add(vals[j])
        # This is O(N * K_max).
        # If K_max is small (e.g., 100), this is fine.
        
        # Let's do that.
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

    except Exception as e:
        # Fallback to smaller N or pure Python if NumPy fails
        pass

    # --- Phase 2: Random Search for Larger N ---
    
    if now() < DEADLINE - 5.0:
        random.seed(12345)
        # Search in [N, 10^12]
        # Sample random starts.
        # To be efficient, we only check if we can beat best_k.
        
        num_trials = 0
        while now() < DEADLINE - 2.0:
            # Pick random start
            # Bias towards larger?
            s = random.randint(N + 1, 10**12 - 200)
            
            # Check run
            seen = set()
            k = 0
            for i in range(200):
                if s + i > 10**12:
                    break
                try:
                    v = phi_large(s + i)
                except:
                    break
                if v in seen:
                    break
                seen.add(v)
                k += 1
            
            if k > best_k:
                best_k = k
                best_s = s
                
            num_trials += 1
            # Throttle?
            # phi_large is fast for most numbers.
            # If k is small, it breaks early.

    return {'start': best_s, 'length': best_k}
