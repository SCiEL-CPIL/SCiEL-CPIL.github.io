"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    deadline = time.time() + 85.0  # 5s safety margin before 90s kill
    
    MAX_N = 1000000
    MIN_N = 10000
    
    # powerful_part[n] = product of p^e for all prime powers p^e || n with e >= 2
    # Use a sieve-like approach: for each prime p, for each power p^2, p^3, ...
    # mark multiples and accumulate the full prime power
    
    powerful = [1] * (MAX_N + 1)
    
    # Sieve of Eratosthenes to get primes
    is_prime = [True] * (MAX_N + 1)
    is_prime[0] = is_prime[1] = False
    primes = []
    for i in range(2, int(MAX_N**0.5) + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, MAX_N + 1, i):
                is_prime[j] = False
    for i in range(int(MAX_N**0.5) + 1, MAX_N + 1):
        if is_prime[i]:
            primes.append(i)
    
    # For each prime, for each exponent e >= 2, compute p^e
    # and for each multiple m of p^e, if p^(e+1) does not divide m,
    # then the contribution is p^e
    # But we need the FULL prime power: if p^e || n, add p^e to powerful[n]
    
    # Better: for each prime p, compute the highest power p^e dividing each multiple
    # and multiply into powerful[n]
    
    for p in primes:
        p2 = p * p
        if p2 > MAX_N:
            break
        # For each multiple of p, find the exact power of p dividing it
        # and if that power >= 2, multiply p^e into powerful[m]
        pe = p2
        # pe = p^2, p^3, ...
        while pe <= MAX_N:
            # For numbers m where p^(e+1) does NOT divide m but p^e divides m
            # i.e., m is divisible by pe but not by pe*p
            pe_next = pe * p
            for m in range(pe, MAX_N + 1, pe):
                if pe_next > MAX_N or m % pe_next != 0:
                    # The exact power of p in m is e (where pe = p^e)
                    powerful[m] *= pe
                    break
            pe = pe_next
    
    # Now find max powerful[n] * powerful[n+1] for n in [MIN_N, MAX_N-1]
    best_score = 0
    best_n = MIN_N
    
    # Check all consecutive pairs
    for n in range(MIN_N, MAX_N):
        score = powerful[n] * powerful[n + 1]
        if score > best_score:
            best_score = score
            best_n = n
    
    if time.time() > deadline:
        return best_n
    
    # Try to improve with targeted B2 construction
    # Look for n where n = a^2 * b and n+1 = c^2 * d with large a^2, c^2
    # Or n = p^2 * s, n+1 = q^2 * t
    
    # Scan for pairs where both have powerful part > 1000
    # and try to find even better ones via local search
    
    # Check around the best found so far with some perturbation
    # Also try constructing: for small primes p, find n ≡ 0 mod p^2 and n+1 ≡ 0 mod q^2
    
    # Use CRT-like approach: n ≡ 0 (mod p^2), n ≡ -1 (mod q^2)
    # n = p^2 * k, and p^2 * k ≡ -1 (mod q^2) => k ≡ -1 * (p^2)^{-1} (mod q^2)
    
    def modinv(a, m):
        # Extended Euclidean
        g, x, _ = _egcd(a, m)
        if g != 1:
            return None
        return x % m
    
    def _egcd(a, b):
        if b == 0:
            return (a, 1, 0)
        g, x1, y1 = _egcd(b, a % b)
        return (g, y1, x1 - (a // b) * y1)
    
    # Try pairs of primes (p, q) with p^2, q^2 up to ~10000
    # n = p^2 * k where k ≡ -inv(p^2, q^2) mod q^2
    # n = p^2 * (k0 + t * q^2)
    
    small_primes = [p for p in primes if p * p <= 100000]
    
    # Limit work
    max_p = 50
    max_q = 50
    
    improved = False
    for i in range(min(max_p, len(small_primes))):
        if time.time() > deadline - 5:
            break
        p = small_primes[i]
        p2 = p * p
        for j in range(min(max_q, len(small_primes))):
            if time.time() > deadline - 5:
                break
            q = small_primes[j]
            if p == q:
                continue
            q2 = q * q
            # n ≡ 0 mod p2, n ≡ -1 mod q2
            # n = p2 * k, p2*k ≡ -1 (mod q2)
            inv = modinv(p2 % q2, q2)
            if inv is None:
                continue
            k0 = (-inv) % q2
            # n = p2 * (k0 + t * q2) for t = 0, 1, 2, ...
            # Find n in [MIN_N, MAX_N-1]
            n_min_t = (MIN_N - p2 * k0 + q2 * p2 - 1) // (q2 * p2)
            n_max_t = (MAX_N - 1 - p2 * k0) // (q2 * p2)
            
            if n_min_t < 0:
                n_min_t = 0
            
            # Check a few values of t
            for t in range(max(0, n_min_t), min(n_max_t + 1, max(0, n_min_t) + 20)):
                if time.time() > deadline - 5:
                    break
                n = p2 * (k0 + t * q2)
                if n < MIN_N or n >= MAX_N:
                    continue
                score = powerful[n] * powerful[n + 1]
                if score > best_score:
                    best_score = score
                    best_n = n
                    improved = True
    
    return best_n
# EVOLVE-BLOCK-END
