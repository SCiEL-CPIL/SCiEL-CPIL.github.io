"""Program using segmented sieve for efficient p-smooth checking of runs after p^2."""
import math
import time

# EVOLVE-BLOCK-START

def is_prime_mr(n):
    """Deterministic Miller-Rabin for n < 3.2e18."""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in witnesses:
        if a >= n:
            continue
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

def sieve_primes(limit):
    """Sieve of Eratosthenes returning list of primes up to limit."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def check_run_segmented(p, primes_small, max_k=200):
    """
    Check the longest run k such that p^2+1, ..., p^2+k are all p-smooth.
    Uses segmented sieve approach: for each number in [p^2+1, p^2+max_k],
    divide out all prime factors from primes_small (primes <= p).
    If remainder is 1 or a prime <= p, the number is p-smooth.
    """
    n_start = p * p + 1
    n_end = p * p + max_k
    
    # Array of remainders for each number in the interval
    remainders = [n_start + i for i in range(max_k)]
    
    # For each small prime q <= p, divide out q from all remainders
    for q in primes_small:
        if q > p:
            break
        # Find first multiple of q in [n_start, n_end]
        first = ((n_start + q - 1) // q) * q
        if first > n_end:
            continue
        for j in range(first - n_start, max_k, q):
            while remainders[j] % q == 0:
                remainders[j] //= q
    
    # Now check each remainder: it must be 1 or a prime <= p
    k = 0
    for i in range(max_k):
        r = remainders[i]
        if r == 1:
            k += 1
            continue
        # r > 1: check if r is prime and <= p
        if r <= p:
            # r is a prime factor <= p (since we divided out all smaller primes)
            k += 1
            continue
        # r > p: check if r is prime
        if is_prime_mr(r):
            # r is a prime > p, so not p-smooth
            break
        # r is composite > p. Since we divided out all primes <= p,
        # r must be a product of primes > p. So not p-smooth.
        break
    
    return k

def construct():
    start_time = time.time()
    time_limit = 115
    
    # Precompute primes up to a reasonable limit for trial division
    # We need primes up to p for the segmented sieve, but we can't store all primes up to 10^9.
    # Strategy: precompute primes up to 10^7. For p > 10^7, we use primes up to 10^7 for division,
    # then check if remainder is prime and <= p.
    SIEVE_LIMIT = 10000000  # 10^7
    primes_small = sieve_primes(SIEVE_LIMIT)
    
    # Also generate primes for candidate p values
    # We'll search primes up to some limit
    MAX_P_SEARCH = 5000000  # Search primes up to 5*10^6
    
    best_k = 0
    best_p = 3
    
    # Generate candidate primes up to MAX_P_SEARCH
    candidate_primes = sieve_primes(MAX_P_SEARCH)
    
    # For each candidate prime p, check the run length
    for p in candidate_primes:
        if time.time() - start_time > time_limit:
            break
        
        # Get primes <= p from our precomputed list
        # Binary search for the index
        lo, hi = 0, len(primes_small)
        while lo < hi:
            mid = (lo + hi) // 2
            if primes_small[mid] <= p:
                lo = mid + 1
            else:
                hi = mid
        primes_up_to_p = primes_small[:lo]
        
        k = check_run_segmented(p, primes_up_to_p)
        if k > best_k:
            best_k = k
            best_p = p
    
    # If we have time, try some larger primes
    if time.time() - start_time < time_limit - 5:
        # Generate some larger prime candidates
        # Use a simple method: check numbers around 5*10^6 to 10^7
        for p_candidate in range(MAX_P_SEARCH + 2, min(SIEVE_LIMIT + 1000, MAX_P_SEARCH + 500000), 2):
            if time.time() - start_time > time_limit:
                break
            if not is_prime_mr(p_candidate):
                continue
            # For these, primes_up_to_p is just primes_small (all <= SIEVE_LIMIT)
            # but we only need those <= p_candidate
            lo, hi = 0, len(primes_small)
            while lo < hi:
                mid = (lo + hi) // 2
                if primes_small[mid] <= p_candidate:
                    lo = mid + 1
                else:
                    hi = mid
            primes_up_to_p = primes_small[:lo]
            
            k = check_run_segmented(p_candidate, primes_up_to_p)
            if k > best_k:
                best_k = k
                best_p = p_candidate
    
    return {'p': best_p, 'k': best_k}

# EVOLVE-BLOCK-END
