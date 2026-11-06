import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 115.0

    # Pre-compute primes up to 10^6 for fast division
    LIMIT = 10**6
    is_prime = [True] * (LIMIT + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(LIMIT**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, LIMIT + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]

    # Precompute phi for all numbers up to LIMIT
    phi_small = list(range(LIMIT + 1))
    for p in primes:
        for j in range(p, LIMIT + 1, p):
            phi_small[j] -= phi_small[j] // p

    # Miller-Rabin for large numbers
    def is_prime_mr(n):
        if n < 2: return False
        # Check small primes first
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n == p: return True
            if n % p == 0: return False
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
            if a >= n: continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    # Pollard's rho for factoring composites
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
        """Full factorization using trial division + Pollard's rho"""
        factors = []
        for p in primes:
            if p * p > n:
                break
            while n % p == 0:
                factors.append(p)
                n //= p
        if n > 1:
            if is_prime_mr(n):
                factors.append(n)
            else:
                # Use Pollard's rho
                stack = [n]
                while stack:
                    m = stack.pop()
                    if m == 1:
                        continue
                    if is_prime_mr(m):
                        factors.append(m)
                    else:
                        d = pollard_rho(m)
                        if d == m:
                            # Retry
                            stack.append(m)
                        else:
                            stack.append(d)
                            stack.append(m // d)
        return factors

    def phi(n):
        if n <= LIMIT:
            return phi_small[n]
        if n == 1:
            return 1
        result = n
        temp = n
        # Divide by small primes
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                result -= result // p
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            if is_prime_mr(temp):
                result -= result // temp
            else:
                # Use Pollard's rho to factor the composite remainder
                facs = factorize(temp)
                for p in set(facs):
                    result -= result // p
        return result

    def check_window(s, max_len=200, max_n=10**12):
        seen = set()
        k = 0
        for i in range(max_len):
            v = s + i
            if v > max_n:
                break
            pv = phi(v)
            if pv in seen:
                break
            seen.add(pv)
            k += 1
        return k

    best_k = 1
    best_s = 1
    
    max_n = 10**12

    # Strategy 1: Fast scan of small starts using precomputed phi
    # Since phi is O(1) for n <= LIMIT, we can scan a large range quickly.
    # We scan up to LIMIT - 200 to ensure we don't exceed the precomputed range
    # without falling back to slow computation (though phi handles it, it's slower).
    # Actually, let's scan up to LIMIT to be safe and fast.
    scan_limit = LIMIT - 200
    for s in range(1, scan_limit + 1):
        if time.time() - start_time > DEADLINE - 20:
            break
        k = check_window(s, 200)
        if k > best_k:
            best_k = k
            best_s = s

    # Strategy 2: Structured search around promising regions in the larger range
    # Try starts at multiples of various values
    # Focus on the region [LIMIT, 10^7] first with a step, then go higher
    # This is a heuristic: runs might be more likely in "smooth" regions or just randomly.
    
    # Let's try a denser search in the range [LIMIT, 10^7]
    # Step size 100 gives 90k checks. Each check involves up to 200 phi calls.
    # If s is in [10^6, 10^7], phi calls are mostly slow (Pollard Rho/Trial).
    # 90k * 200 = 18M phi calls. This might be too slow.
    # Let's estimate phi speed. ~0.1ms per phi? 18M * 0.1ms = 1800s. Too slow.
    # So we need to be careful with the larger range.
    
    # Let's do a sparse scan in the larger range.
    # Step 1000 in [10^6, 10^8] -> 100k checks. Still too many if each is slow.
    
    # Let's refine Strategy 2:
    # Use a larger step for the larger range.
    # Step 1000 in [10^6, 10^7] -> 9000 checks.
    # 9000 * 200 = 1.8M phi calls. ~180s. Too slow.
    # Step 10000 in [10^6, 10^8] -> 9000 checks.
    
    # Let's try a mix.
    # First, check specific promising spots? No, random is better.
    
    # Strategy 2: Random search with local refinement
    random.seed(42)
    remaining_time = DEADLINE - (time.time() - start_time)
    
    # We have roughly 20-30 seconds left after Strategy 1.
    # Let's do random sampling in [1, max_n].
    # But bias towards smaller numbers? Or uniform?
    # The task says "Long runs below 10^12 are finite evidence".
    # Smaller numbers have more constraints (phi(n) is smaller, more collisions).
    # Larger numbers have larger phi(n), fewer collisions?
    # Actually, phi(n) ~ n * prod(1-1/p).
    # For random n, phi(n) is unique with high probability for large n?
    # Collisions occur when phi(a) = phi(b).
    # For large n, the density of values is higher, so collisions are less likely?
    # Wait, phi(n) < n. The range of phi values is [1, n].
    # For a window of length k, we have k values in [1, n].
    # If n is large, the values are spread out.
    # If n is small, values are compressed.
    # So larger s should have longer runs on average.
    
    # Therefore, we should bias the random search towards larger s.
    
    for _ in range(5000):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Sample s from [1, max_n]
        # To bias towards larger numbers, we can sample log-uniformly?
        # Or just uniform.
        # Let's try uniform first.
        s = random.randint(1, max_n - 200)
        
        k = check_window(s, 200)
        if k > best_k:
            best_k = k
            best_s = s
            
        # Local refinement around promising spots
        if k > 10:
            # Check neighbors
            for offset in range(-20, 21, 2):
                if time.time() - start_time > DEADLINE - 5:
                    break
                ns = s + offset
                if 1 <= ns <= max_n - 200:
                    nk = check_window(ns, 200)
                    if nk > best_k:
                        best_k = nk
                        best_s = ns

    return {'start': best_s, 'length': best_k}
