import math
import time

# Precompute small primes for trial division
_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]

# Deterministic Miller-Rabin bases for n < 3,317,044,064,679,887,385,961,981
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return n in (2, 3)
    for p in _SMALL_PRIMES:
        if n == p:
            return True
        if p * p > n:
            break
        if n % p == 0:
            return False
    # Miller-Rabin
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MR_BASES:
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

def pollard_brent(n, seed=1):
    """Brent's variant of Pollard's rho - faster than basic rho."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    if n % 7 == 0:
        return 7
    if n % 11 == 0:
        return 11
    if n % 13 == 0:
        return 13
    
    y = seed % n
    c = (seed * 7 + 13) % n
    m = 128
    g, r, q = 1, 1, 1
    while g == 1:
        x = y
        for _ in range(r):
            y = (y * y + c) % n
        k = 0
        while k < r and g == 1:
            ys = y
            for _ in range(min(m, r - k)):
                y = (y * y + c) % n
                q = (q * abs(x - y)) % n
            g = math.gcd(q, n)
            k += m
        r *= 2
        if r > 1000000:  # Safety break
            break
    if g == n:
        while True:
            ys = (ys * ys + c) % n
            g = math.gcd(abs(x - ys), n)
            if g > 1:
                break
        if g == n:
            # Retry with different seed
            return pollard_brent(n, seed + 1)
    return g

def largest_prime_factor(n, p_limit):
    """Find largest prime factor of n. Return early if factor > p_limit."""
    if n == 1:
        return 1
    if n < 0:
        n = -n
    # Trial division by small primes
    max_f = 1
    for p in _SMALL_PRIMES:
        if p > p_limit:
            break
        if p * p > n:
            break
        if n % p == 0:
            max_f = p
            while n % p == 0:
                n //= p
    if n == 1:
        return max_f
    if is_prime(n):
        if n > p_limit:
            return n
        return max(n, max_f)
    # Recursively find factors
    stack = [n]
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if is_prime(m):
            if m > p_limit:
                return m
            if m > max_f:
                max_f = m
        else:
            d = pollard_brent(m)
            if d == m:
                # Shouldn't happen, but handle it
                if m > p_limit:
                    return m
                if m > max_f:
                    max_f = m
            else:
                stack.append(d)
                stack.append(m // d)
    return max_f

def is_p_smooth(n, p):
    """Check if n is p-smooth (all prime factors <= p)."""
    if n == 1:
        return True
    if n < 0:
        n = -n
    # Trial division by small primes
    for prime in _SMALL_PRIMES:
        if prime > p:
            break
        if prime * prime > n:
            break
        if n % prime == 0:
            while n % prime == 0:
                n //= prime
    if n == 1:
        return True
    if is_prime(n):
        return n <= p
    # Recursively check factors
    stack = [n]
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if is_prime(m):
            if m > p:
                return False
        else:
            d = pollard_brent(m)
            if d == m:
                if m > p:
                    return False
            else:
                stack.append(d)
                stack.append(m // d)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    best_k = 0
    best_p = 2
    
    # Sieve primes up to 10^7
    limit = 10**7
    is_prime_sieve = bytearray(b'\x01') * (limit + 1)
    is_prime_sieve[0] = False
    is_prime_sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime_sieve[i]:
            is_prime_sieve[i*i::i] = bytearray(len(is_prime_sieve[i*i::i]))
    
    primes = [i for i in range(2, limit + 1) if is_prime_sieve[i]]
    
    # Check each prime p
    for p in primes:
        if time.time() > deadline:
            break
        
        p2 = p * p
        
        # Quick rejection: if p^2+1 is prime and > p, k=0
        n1 = p2 + 1
        if is_prime(n1):
            if n1 > p:
                continue
        
        k = 0
        for i in range(1, 201):
            if time.time() > deadline:
                break
                
            n_val = p2 + i
            
            # Quick check: is n_val prime? If so and n_val > p, fail
            if is_prime(n_val):
                if n_val > p:
                    break
                else:
                    k += 1
                    continue
            
            # Find largest prime factor
            lf = largest_prime_factor(n_val, p)
            
            if lf > p:
                break
            
            k += 1
            
        if k > best_k:
            best_k = k
            best_p = p
    
    # Try some larger primes beyond the sieve
    if time.time() < deadline - 5:
        import random
        random.seed(42)
        for _ in range(200):
            if time.time() > deadline:
                break
            p_candidate = random.randint(10**7, 10**9)
            if p_candidate % 2 == 0:
                p_candidate += 1
            if is_prime(p_candidate):
                p = p_candidate
                p2 = p * p
                n1 = p2 + 1
                if is_prime(n1):
                    if n1 > p:
                        continue
                k = 0
                for i in range(1, 201):
                    if time.time() > deadline:
                        break
                    n_val = p2 + i
                    if is_prime(n_val):
                        if n_val > p:
                            break
                        else:
                            k += 1
                            continue
                    lf = largest_prime_factor(n_val, p)
                    if lf > p:
                        break
                    k += 1
                if k > best_k:
                    best_k = k
                    best_p = p

    return {'p': best_p, 'k': best_k}
