import math
import time

# Precompute small primes for trial division and Miller-Rabin
_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

def is_prime(n):
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return False
    # Miller-Rabin
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    # For n < 3,825,123,056,546,413,051, first 12 primes suffice
    for a in _SMALL_PRIMES:
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
    if n % 5 == 0:
        return 5
    x = 2
    y = 2
    d = 1
    c = 1
    while d == 1:
        x = (x * x + c) % n
        y = (y * y + c) % n
        y = (y * y + c) % n
        d = math.gcd(abs(x - y), n)
        if d == n:
            c += 1
            x = 2
            y = 2
            d = 1
            if c > 10:
                x = 3
                y = 3
                c = 1
    return d

def is_p_smooth(n, p, primes_up_to_p):
    """Check if n is p-smooth using precomputed primes up to min(p, max(primes_up_to_p))"""
    if n <= 1:
        return True
    for q in primes_up_to_p:
        if q > p:
            break
        if q * q > n:
            break
        if n % q == 0:
            while n % q == 0:
                n //= q
    if n == 1:
        return True
    # n is now either prime or has a prime factor > p
    if is_prime(n):
        return n <= p
    # n is composite with all prime factors > sqrt(n) or mixed
    # Use Pollard's rho to factor
    d = pollard_rho(n)
    if d == n:
        # Pollard failed, try again with different seed
        d = pollard_rho(n)
    if d == n:
        # Still failed, check if n itself is <= p (shouldn't happen for composite)
        return n <= p
    f1 = d
    f2 = n // d
    # Recursively check both factors
    return is_p_smooth(f1, p, primes_up_to_p) and is_p_smooth(f2, p, primes_up_to_p)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    best_k = 0
    best_p = 2

    # Sieve primes up to 10^6 for trial division in smoothness check
    # This gives us ~78k primes, which is manageable
    sieve_limit = 10**6
    is_prime_sieve = bytearray(b'\x01') * (sieve_limit + 1)
    is_prime_sieve[0] = False
    is_prime_sieve[1] = False
    for i in range(2, int(sieve_limit**0.5) + 1):
        if is_prime_sieve[i]:
            is_prime_sieve[i*i::i] = bytearray(len(is_prime_sieve[i*i::i]))
    
    primes_list = [i for i in range(2, sieve_limit + 1) if is_prime_sieve[i]]
    
    # Also sieve candidate primes up to 5*10^7 to have good candidates
    # 5*10^7 / ln(5*10^7) ≈ 3M primes
    cand_limit = 5 * 10**7
    is_prime_cand = bytearray(b'\x01') * (cand_limit + 1)
    is_prime_cand[0] = False
    is_prime_cand[1] = False
    for i in range(2, int(cand_limit**0.5) + 1):
        if is_prime_cand[i]:
            is_prime_cand[i*i::i] = bytearray(len(is_prime_cand[i*i::i]))
    
    candidate_primes = [i for i in range(2, cand_limit + 1) if is_prime_cand[i]]

    # For each candidate prime p, check smoothness of p^2+1, ..., p^2+200
    # Use the precomputed primes_list for trial division in is_p_smooth
    
    for p in candidate_primes:
        if time.time() > deadline:
            break
        
        # Quick rejection: if p is very small, skip
        if p < 7:
            continue
        
        # Check sequence
        k = 0
        p_sq = p * p
        
        for i in range(1, 201):
            if time.time() > deadline:
                break
                
            n_val = p_sq + i
            
            # Quick check: is n_val prime? If so and n_val > p, fail
            if is_prime(n_val):
                if n_val > p:
                    break
                else:
                    k += 1
                    continue
            
            # Check if p-smooth
            if not is_p_smooth(n_val, p, primes_list):
                break
            else:
                k += 1
        
        if k > best_k:
            best_k = k
            best_p = p

    # If we have time, try some larger primes beyond the sieve
    if time.time() < deadline - 20:
        import random
        random.seed(42)
        for _ in range(100):
            if time.time() > deadline:
                break
            p_candidate = random.randint(10**7, 10**9)
            if p_candidate % 2 == 0:
                p_candidate += 1
            if is_prime(p_candidate):
                p = p_candidate
                k = 0
                p_sq = p * p
                n1 = p_sq + 1
                if is_prime(n1):
                    if n1 > p:
                        continue
                for i in range(1, 201):
                    if time.time() > deadline:
                        break
                    n_val = p_sq + i
                    if is_prime(n_val):
                        if n_val > p:
                            break
                        else:
                            k += 1
                            continue
                    if not is_p_smooth(n_val, p, primes_list):
                        break
                    else:
                        k += 1
                if k > best_k:
                    best_k = k
                    best_p = p

    return {'p': best_p, 'k': best_k}
