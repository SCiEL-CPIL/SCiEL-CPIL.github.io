"""
Strategy: Structured Grid Search on Exponents for Small Prime Sets.
Instead of random sampling of n, we systematically iterate over exponent tuples
(a, b, ...) for small primes (2, 3, 5, ...) to find the configuration that
maximizes the number of coprime adjacent divisor pairs.
"""
# EVOLVE-BLOCK-START
import time
import math
import heapq

DEADLINE = 145.0
_SMALL_PRIMES = [2, 3, 5, 7, 11, 13]

def _score(n):
    """
    Calculate the number of coprime adjacent pairs in the sorted divisors of n.
    """
    if n < 2:
        return 0
    
    # Generate divisors
    # Factorize n using small primes
    divisors = [1]
    temp = n
    for p in _SMALL_PRIMES:
        if p * p > temp:
            break
        if temp % p == 0:
            power = 1
            while temp % p == 0:
                temp //= p
                power *= p
                divisors += [d * power for d in divisors]
    if temp > 1:
        divisors += [d * temp for d in divisors]
    
    divisors = sorted(set(divisors))
    
    count = 0
    for i in range(1, len(divisors)):
        if math.gcd(divisors[i-1], divisors[i]) == 1:
            count += 1
    return count

def construct():
    start = time.time()
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19
    best_s = _score(best_n)
    
    def consider(n):
        nonlocal best_n, best_s
        if 2 <= n <= 10**18:
            s = _score(n)
            if s > best_s:
                best_s, best_n = s, n
                return True
        return False

    # 1. Search 2-prime numbers: n = p^a * q^b
    for i, p in enumerate(_SMALL_PRIMES):
        for q in _SMALL_PRIMES[i+1:]:
            if p * q > 10**18:
                continue
            # Determine max exponents
            max_a = int(math.log(10**18 // q, p)) + 1
            max_b = int(math.log(10**18 // p, q)) + 1
            
            # Iterate exponents
            # To save time, we can restrict the search space if it's too large
            # But for p=2, q=3, max_a ~ 60, max_b ~ 37. 60*37 = 2220 checks.
            # Each check is O(tau(n) log tau(n)).
            # For 2^30 * 3^20, tau = 31*21 = 651. Sort 651 ints. Fast.
            for a in range(1, max_a + 1):
                pa = p ** a
                if pa > 10**18:
                    break
                rem = 10**18 // pa
                max_b_local = int(math.log(rem, q)) + 1
                for b in range(1, max_b_local + 1):
                    n = pa * (q ** b)
                    if n > 10**18:
                        break
                    if time.time() > DEADLINE:
                        return best_n
                    consider(n)
                    # Also check b=0? No, a=0 is covered by p loop start? 
                    # We start a=1. What about n=p? 
                    # consider(p) is covered if we allow b=0? 
                    # Let's include 0 in range to be safe, but n>=2.
                    # Actually, range(1, ...) misses n=p (b=0) and n=q (a=0).
                    # Let's adjust ranges to include 0, but skip n=1.

    # Re-run with 0 included for completeness? 
    # The loop above started at 1. Let's add a pass for single primes.
    for p in _SMALL_PRIMES:
        consider(p)
        
    # 2. Search 3-prime numbers: n = p^a * q^b * r^c
    # This is more expensive. 
    # Limit to small primes 2, 3, 5.
    p, q, r = 2, 3, 5
    max_a = int(math.log(10**18 // (q*r), p)) + 1
    for a in range(1, max_a + 1):
        if time.time() > DEADLINE:
            return best_n
        pa = p ** a
        rem = 10**18 // pa
        max_b = int(math.log(rem // r, q)) + 1
        for b in range(1, max_b + 1):
            pq = pa * (q ** b)
            if pq > 10**18:
                break
            rem2 = 10**18 // pq
            max_c = int(math.log(rem2, r)) + 1
            for c in range(1, max_c + 1):
                n = pq * (r ** c)
                if n > 10**18:
                    break
                if time.time() > DEADLINE:
                    return best_n
                consider(n)
    
    # 3. Search 4-prime numbers: 2, 3, 5, 7
    # Only if time permits
    if time.time() < DEADLINE - 30:
        p, q, r, s = 2, 3, 5, 7
        max_a = int(math.log(10**18 // (q*r*s), p)) + 1
        for a in range(1, max_a + 1):
            if time.time() > DEADLINE:
                return best_n
            pa = p ** a
            rem = 10**18 // pa
            max_b = int(math.log(rem // (r*s), q)) + 1
            for b in range(1, max_b + 1):
                if time.time() > DEADLINE:
                    return best_n
                pq = pa * (q ** b)
                if pq > 10**18:
                    break
                rem2 = 10**18 // pq
                max_c = int(math.log(rem2 // s, r)) + 1
                for c in range(1, max_c + 1):
                    if time.time() > DEADLINE:
                        return best_n
                    pqr = pq * (r ** c)
                    if pqr > 10**18:
                        break
                    rem3 = 10**18 // pqr
                    max_d = int(math.log(rem3, s)) + 1
                    for d in range(1, max_d + 1):
                        n = pqr * (s ** d)
                        if n > 10**18:
                            break
                        if time.time() > DEADLINE:
                            return best_n
                        consider(n)

    return best_n
# EVOLVE-BLOCK-END
