"""CRT-based construction for maximizing powerful_part(n)*powerful_part(n+1)."""
import time
from math import gcd, isqrt

def _egcd(a, b):
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = _egcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)

def _modinv(a, m):
    if m == 1:
        return 0
    g, x, _ = _egcd(a, m)
    if g != 1:
        return None
    return x % m

def _sieve_spf(limit):
    """Smallest prime factor sieve."""
    spf = list(range(limit + 1))
    if limit >= 0:
        spf[0] = 0
    if limit >= 1:
        spf[1] = 1
    for i in range(2, isqrt(limit) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def _powerful_part_spf(n, spf):
    """Compute powerful part: product of p^e for primes with e>=2."""
    if n <= 1:
        return 1
    pp = 1
    while n > 1:
        p = spf[n]
        e = 0
        while n % p == 0:
            n //= p
            e += 1
        if e >= 2:
            pp *= p ** e
    return pp

def _generate_powerful_numbers(limit, spf):
    """
    Generate all powerful numbers <= limit.
    A powerful number has all prime exponents >= 2.
    Return sorted list of powerful numbers.
    """
    powerful = set()
    # For each prime p, generate p^2, p^3, ... <= limit
    prime_powers = []
    for p in range(2, limit + 1):
        if spf[p] == p:  # prime
            pw = p * p
            powers = []
            while pw <= limit:
                powers.append(pw)
                pw *= p
            if powers:
                prime_powers.append((p, powers))
    
    # DFS to combine distinct prime powers
    def build(idx, current):
        if current > limit:
            return
        if idx == len(prime_powers):
            if current > 1:
                powerful.add(current)
            return
        # skip this prime
        build(idx + 1, current)
        # include each power of this prime
        p, powers = prime_powers[idx]
        for pw in powers:
            if current * pw > limit:
                break
            build(idx + 1, current * pw)
    
    build(0, 1)
    return sorted(powerful)

def construct():
    start = time.time()
    DEADLINE = 75  # safety margin before 90s kill
    
    def time_left():
        return DEADLINE - (time.time() - start)
    
    # Precompute SPF
    spf = _sieve_spf(1000000)
    
    # Generate powerful numbers
    pn_list = _generate_powerful_numbers(1000000, spf)
    
    best_n = 10000
    best_score = _powerful_part_spf(10000, spf) * _powerful_part_spf(10001, spf)
    
    # CRT-based search: for pairs of powerful numbers (A, C) with gcd(A,C)=1,
    # solve n ≡ 0 (mod A), n ≡ -1 (mod C)
    # Then n = A*t0 + k*A*C, check if in range and compute actual score
    
    # Filter to powerful numbers >= 4
    pn_filtered = [p for p in pn_list if p >= 4]
    
    # Limit pair product to keep solutions in range
    MAX_PRODUCT = 3_000_000
    
    checked = 0
    for i, A in enumerate(pn_filtered):
        if time_left() < 5:
            break
        for C in pn_filtered:
            if time_left() < 5:
                break
            if A * C > MAX_PRODUCT:
                continue
            if gcd(A, C) != 1:
                continue
            # n = A*t, A*t ≡ -1 (mod C) => t ≡ -A^{-1} (mod C)
            inv = _modinv(A, C)
            if inv is None:
                continue
            t0 = (-inv) % C
            n0 = A * t0
            M = A * C
            # n = n0 + k*M in [10000, 1000000]
            k_min = max(0, (10000 - n0 + M - 1) // M)
            k_max = (1000000 - n0) // M
            if k_max < k_min:
                continue
            # Check k values
            max_k_check = min(k_max + 1, k_min + 20)
            for k in range(k_min, max_k_check):
                n = n0 + k * M
                if n < 10000 or n > 1000000:
                    continue
                pp_n = _powerful_part_spf(n, spf)
                pp_np1 = _powerful_part_spf(n + 1, spf)
                score = pp_n * pp_np1
                if score > best_score:
                    best_score = score
                    best_n = n
                checked += 1
                if checked % 5000 == 0 and time_left() < 10:
                    break
    
    # Direct scan with spf for remaining time
    if time_left() > 8:
        step = 1
        if time_left() < 25:
            step = 5
        if time_left() < 15:
            step = 10
        if time_left() < 10:
            step = 25
        for n in range(10000, 1000001, step):
            if time_left() < 3:
                break
            pp_n = _powerful_part_spf(n, spf)
            pp_np1 = _powerful_part_spf(n + 1, spf)
            score = pp_n * pp_np1
            if score > best_score:
                best_score = score
                best_n = n
    
    if best_n < 10000 or best_n > 1000000:
        best_n = 10000
    
    return best_n
