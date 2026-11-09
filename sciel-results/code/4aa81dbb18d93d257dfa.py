"""Program for: Long runs after p^2 with no prime factor exceeding p (p <= 10^9).
Uses a direct approach: search for primes p and check consecutive integers p^2+1, ..., p^2+k
for p-smoothness (all prime factors <= p). Uses efficient trial division and Pollard-Brent
with deterministic Miller-Rabin for factorization. Maximizes k within time budget."""

import time
import math
import random

# ---------- Deterministic Miller-Rabin for n < 3.3e24 ----------
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

def is_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MR_BASES:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True

# ---------- Pollard's Brent ----------
def pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    while True:
        y = random.randrange(1, n)
        c = random.randrange(1, n)
        m = 128
        g = r = q = 1
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = q * abs(x - y) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
                if g != 1:
                    break
        if g != n:
            return g

# ---------- Precompute small primes for trial division ----------
_SMALL_PRIMES = []
def _init_small_primes(limit=10000):
    """Generate small primes for fast trial division."""
    if limit < 2:
        return
    is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    _SMALL_PRIMES.extend(i for i in range(2, limit + 1) if is_p[i])

_init_small_primes(10000)

def is_p_smooth(n, p):
    """Check if all prime factors of n are <= p. Returns True if n is 1 or p-smooth."""
    if n <= 1:
        return True
    
    # Try small primes first with trial division
    for q in _SMALL_PRIMES:
        if q > p:
            break
        while n % q == 0:
            n //= q
        if n == 1:
            return True
    
    if n == 1:
        return True
    
    # If n is prime, check if it's <= p
    if is_prime(n):
        return n <= p
    
    # Use Pollard-Brent to find a factor, then recursively check
    # We need to find all factors and check each is <= p
    factors = []
    _factor(n, factors, p)
    for f in factors:
        if f > p:
            return False
    return True

def build_presieve_mask(p_max, k_max=200, q_limit=50000):
    """
    Build a bitset marking which residues r mod q are 'bad' for each small prime q.
    A residue r is bad if for some j in [1, k_max], r^2 + j ≡ 0 (mod q).
    This lets us quickly eliminate candidate p values without factorization.
    
    Returns: dict mapping q -> set of bad residues, or None if too expensive.
    """
    if p_max < 2:
        return {}
    
    bad_residues = {}
    # We only need to consider primes q that are > p_max (those <= p_max are already handled by trial division)
    # But actually, for the presieve to be useful, we want q values where p^2 + j can be divisible by q
    # for some j in [1, k_max]. If q <= p_max, then q-smoothness already handles it.
    # So we focus on q > p_max.
    # However, generating primes up to q_limit and filtering those > p_max could be expensive.
    # Let's use a different approach: for each small prime q (from our _SMALL_PRIMES list),
    # compute which residues r mod q make p^2 + j ≡ 0 (mod q) for some j in [1, k_max].
    # But wait - if q <= p, then q is already a valid factor. The issue is when q > p.
    # So the presieve should focus on primes q > p.
    
    # Actually, let me reconsider. The presieve works as follows:
    # For a candidate p, if there exists a prime q > p such that q | (p^2 + j) for some j in [1, k_max],
    # then p^2 + j has a prime factor q > p, so p is NOT k_max-smooth for that j.
    # So we want to eliminate p values where such a q exists.
    # 
    # For each prime q, the bad residues are those r where r^2 ≡ -j (mod q) for some j in [1, k_max].
    # We can compute this for each q and mark p values accordingly.
    
    # Generate primes up to q_limit
    q_primes = sieve_primes(q_limit)
    
    for q in q_primes:
        # Compute bad residues: r such that r^2 + j ≡ 0 (mod q) for some j in [1, k_max]
        bad = set()
        for j in range(1, k_max + 1):
            target = (-j) % q
            # Solve r^2 ≡ target (mod q)
            # Try all residues
            for r in range(q):
                if (r * r) % q == target:
                    bad.add(r)
                    break  # No need to continue for this j if we found one r
        # Actually, we need ALL r such that for SOME j, r^2 ≡ -j (mod q)
        # Let me redo this
        bad = set()
        for r in range(q):
            r2 = (r * r) % q
            for j in range(1, k_max + 1):
                if (r2 + j) % q == 0:
                    bad.add(r)
                    break
        if bad:
            bad_residues[q] = bad
    
    return bad_residues

def _factor(n, factors, p):
    """Recursively factor n, adding factors to list. Early termination if factor > p."""
    if n == 1:
        return
    if is_prime(n):
        factors.append(n)
        return
    d = pollard_brent(n)
    _factor(d, factors, p)
    _factor(n // d, factors, p)

def sieve_primes(limit):
    """Generate all primes up to limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    return [i for i in range(2, limit + 1) if is_p[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s kill
    
    best_k = 0
    best_p = 2
    
    # Generate primes up to a reasonable limit using sieve
    sieve_limit = 5_000_000
    primes = sieve_primes(sieve_limit)
    
    # Build a presieve mask for quick elimination
    # For each small prime q, determine which residues r mod q are "bad"
    # (i.e., r^2 + j ≡ 0 mod q for some j in [1, 200])
    # If p ≡ r (mod q) for a bad residue, then p^2 + j is divisible by q for some j,
    # and if q > p, this means p^2 + j has a prime factor > p.
    
    # We'll use a segmented approach: for a range of p values, mark which ones are eliminated
    # by the presieve, then only do full factorization on survivors.
    
    # First, build the presieve data structure
    # For efficiency, we'll work with primes q up to some limit
    presieve_q_limit = 10000
    presieve_q_primes = [q for q in _SMALL_PRIMES if q <= presieve_q_limit]
    
    # For each q, compute bad residues
    presieve_data = []  # list of (q, bad_residue_set)
    for q in presieve_q_primes:
        bad = set()
        for r in range(q):
            r2 = (r * r) % q
            for j in range(1, 201):
                if (r2 + j) % q == 0:
                    bad.add(r)
                    break
        if bad:
            presieve_data.append((q, bad))
    
    def is_presieved(p, presieve_data):
        """Check if p is eliminated by the presieve.
        Returns True if p should be skipped (has a prime factor > p in some p^2+j)."""
        for q, bad in presieve_data:
            if q <= p:
                continue  # q <= p is fine, q is a valid factor
            if p % q in bad:
                return True
        return False
    
    # Strategy: 
    # 1. Try all primes up to sieve_limit
    # 2. For each, first check presieve; if eliminated, skip
    # 3. Otherwise, compute how far the run goes
    # 4. Keep track of best
    
    for p in primes:
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        # Quick presieve check
        if is_presieved(p, presieve_data):
            continue
        
        k = 0
        p_sq = p * p
        while k < 200:
            if time.time() - start_time > deadline:
                break
            n = p_sq + k + 1
            if not is_p_smooth(n, p):
                break
            k += 1
        
        if k > best_k:
            best_k = k
            best_p = p
    
    # If time remains, try some larger primes with better heuristics
    if time.time() - start_time < deadline - 20:
        # Try primes in ranges larger than sieve_limit
        lo = max(sieve_limit + 1, 10**6)
        hi = min(10**9, 10**8)
        
        # Try structured search: sample primes in geometric progression ranges
        num_samples = 500
        for i in range(num_samples):
            if time.time() - start_time > deadline - 5:
                break
            
            # Interpolate between lo and hi with some randomness
            frac = i / num_samples
            base_lo = int(lo * (1 + frac * 0.5))
            base_hi = min(hi, int(base_lo * 1.5))
            
            candidate = random.randrange(base_lo, base_hi + 1)
            if candidate % 2 == 0:
                candidate += 1
            
            # Find next prime
            found_prime = False
            for _ in range(100):
                if is_prime(candidate):
                    found_prime = True
                    break
                candidate += 2
                if candidate > base_hi:
                    break
            
            if not found_prime:
                continue
            
            # Presieve check
            if is_presieved(candidate, presieve_data):
                continue
            
            k = 0
            p_sq = candidate * candidate
            while k < 200:
                if time.time() - start_time > deadline - 1:
                    break
                n = p_sq + k + 1
                if not is_p_smooth(n, candidate):
                    break
                k += 1
            
            if k > best_k:
                best_k = k
                best_p = candidate
    
    return {'p': best_p, 'k': best_k}
