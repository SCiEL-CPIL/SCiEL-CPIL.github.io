"""Constructive search for long runs of distinct totients using fast factorization."""
# EVOLVE-BLOCK-START
import random
import math

def construct():
    MAX_N = 10**12
    MAX_K = 200
    
    # Precompute primes up to 10^6 for trial division
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    
    # Miller-Rabin primality test
    def is_probable_prime(n, k=10):
        if n < 2:
            return False
        # Small primes
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                return n == p
        # Write n-1 as d*2^r
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        # Witnesses
        for _ in range(k):
            a = random.randrange(2, n - 1)
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

    # Pollard Rho
    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        while True:
            c = random.randrange(1, n)
            x = random.randrange(2, n)
            y = x
            d = 1
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d

    def factorize(n):
        """Return list of prime factors (with multiplicity) of n."""
        factors = []
        for p in primes:
            if p * p > n:
                break
            while n % p == 0:
                factors.append(p)
                n //= p
        if n > 1:
            if is_probable_prime(n):
                factors.append(n)
            else:
                # n is composite, likely product of two large primes
                # Use Pollard Rho to split
                d = pollard_rho(n)
                factors.extend(factorize(d))
                factors.extend(factorize(n // d))
        return factors

    def phi(n):
        """Compute Euler's totient function."""
        if n == 1:
            return 1
        original_n = n
        result = n
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
        if n > 1:
            # n is prime
            result -= result // n
        return result

    best_k = 1
    best_s = 1
    
    # Candidate generation strategies
    # 1. Multiples of primorials
    # 2. Random high values
    
    # Primorial P_10 = 2*3*5*7*11*13*17*19*23*29 = 6469693230
    P_10 = 6469693230
    P_9 = P_10 // 29
    P_8 = P_9 // 23
    
    candidates = set()
    
    # Add candidates around primorials
    for P in [P_8, P_9, P_10]:
        for k_mult in range(1, 1000):
            s = k_mult * P
            if s < 10**10:
                continue
            if s + MAX_K > MAX_N:
                break
            candidates.add(s)
            candidates.add(s + 1)
            candidates.add(s - 1)
            
    # Add random candidates in high range
    for _ in range(2000):
        s = random.randint(10**11, MAX_N - MAX_K)
        candidates.add(s)
        
    # Sort candidates to check systematically
    candidates = sorted(candidates)
    
    for s in candidates:
        if s < 1 or s + MAX_K > MAX_N:
            continue
            
        # Compute phi for s .. s+MAX_K-1
        phis = []
        for i in range(MAX_K):
            n = s + i
            if n > MAX_N:
                break
            phis.append(phi(n))
            
        # Find longest distinct run
        seen = set()
        left = 0
        for right in range(len(phis)):
            v = phis[right]
            if v in seen:
                while left <= right:
                    seen.discard(phis[left])
                    left += 1
                    if left > right:
                        break
            seen.add(v)
            cur_len = right - left + 1
            if cur_len > best_k:
                best_k = cur_len
                best_s = s + left
                
    return {'start': int(best_s), 'length': int(best_k)}
# EVOLVE-BLOCK-END
