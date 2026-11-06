import time
import math
import random

def construct():
    start_time = time.time()
    deadline = 120  # seconds, safety margin

    # Deterministic Miller-Rabin for n < 3.3 * 10^24
    # Witnesses: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    def is_prime(n):
        if n < 2:
            return False
        # Small primes
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
        # Test witnesses
        for a in small_primes:
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

    # Sieve primes up to a limit
    def sieve(limit):
        if limit < 2:
            return []
        is_p = bytearray([1]) * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit ** 0.5) + 1):
            if is_p[i]:
                for j in range(i * i, limit + 1, i):
                    is_p[j] = 0
        return [i for i in range(2, limit + 1) if is_p[i]]

    # Generate primes up to p using segmented sieve for large p
    def generate_primes_up_to(p):
        if p < 2:
            return []
        if p < 100000:
            return sieve(p)
        # Use segmented sieve for larger p
        primes_small = sieve(10000)
        limit = int(math.isqrt(p))
        # Get primes up to sqrt(p) from small primes
        primes_for_sieve = [q for q in primes_small if q <= limit]
        
        # Segmented sieve from 10000 to p
        low = 10000
        high = p
        size = high - low + 1
        is_prime_seg = bytearray([1]) * size
        
        for q in primes_for_sieve:
            start = max(q * q, ((low + q - 1) // q) * q)
            for j in range(start - low, size, q):
                is_prime_seg[j] = 0
        
        result = primes_small[:]
        for i in range(low, high + 1):
            if is_prime_seg[i - low]:
                result.append(i)
        return result

    # For a given prime p, check the run length k using segmented factorization
    # We factor p^2+1, ..., p^2+200 using all primes <= p
    def check_run_length(p, max_k=200):
        p2 = p * p
        
        # We need to factor n = p^2 + i for i = 1..max_k
        # All prime factors must be <= p
        # Strategy: divide out all primes <= p from each number
        # If the remaining factor is 1, it's good; otherwise it has a prime factor > p
        
        # Get all primes <= p
        # For efficiency, we only need primes that could divide numbers in [p^2+1, p^2+max_k]
        # A prime q <= p divides p^2+i if p^2+i ≡ 0 (mod q), i.e., i ≡ -p^2 (mod q)
        # So for each prime q <= p, we can find which i in [1, max_k] are divisible by q
        
        # Generate primes up to p
        if p <= 100000:
            primes = sieve(p)
        else:
            primes = generate_primes_up_to(p)
        
        # For each number p^2 + i, track its remaining factor after dividing by all primes <= p
        # rem[i] = (p^2 + i) with all prime factors <= p removed
        # If rem[i] == 1, then p^2+i is p-smooth
        
        rem = [1] * (max_k + 1)  # rem[i] for i = 1..max_k, rem[0] unused
        for i in range(1, max_k + 1):
            rem[i] = p2 + i
        
        # For each prime q <= p, find multiples in [p^2+1, p^2+max_k]
        # p^2 + i ≡ 0 (mod q) => i ≡ -p^2 (mod q)
        # Let r = (-p^2) mod q
        # Then i = r, r+q, r+2q, ... within [1, max_k]
        
        for q in primes:
            # Find i in [1, max_k] such that p^2 + i ≡ 0 (mod q)
            # i ≡ -p^2 (mod q)
            r = (-p2) % q
            if r == 0:
                r = q  # i must be >= 1
            
            # i values: r, r+q, r+2q, ... <= max_k
            i = r
            while i <= max_k:
                # Divide rem[i] by q as many times as possible
                while rem[i] % q == 0:
                    rem[i] //= q
                i += q
        
        # Now find the longest run from i=1 where rem[i] == 1
        k = 0
        for i in range(1, max_k + 1):
            if rem[i] == 1:
                k += 1
            else:
                break
        
        return k

    best_k = 0
    best_p = 2

    # Initial small check
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
        k = check_run_length(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Search larger primes in batches
    # We'll try primes up to 10^6, then 10^7 if time permits
    search_limits = [10**4, 10**5, 10**6, 10**7]
    prev_limit = 50
    
    for limit in search_limits:
        if time.time() - start_time > deadline - 15:
            break
        
        # Get primes in (prev_limit, limit]
        primes = sieve(limit)
        for p in primes:
            if p <= prev_limit:
                continue
            if time.time() - start_time > deadline - 5:
                break
            
            # Quick check: if p is small, we already checked
            k = check_run_length(p)
            if k > best_k:
                best_k = k
                best_p = p
        
        prev_limit = limit

    return {'p': best_p, 'k': best_k}

if __name__ == "__main__":
    res = construct()
    print(res)
