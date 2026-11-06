import time, random

_START = time.time()
_DEADLINE = _START + 118.0  # safety margin under 120s kill

def _time_ok():
    return time.time() < _DEADLINE

# --- deterministic Miller-Rabin for 64-bit ---
_MR_BASIS = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

def is_prime(n):
    if n < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small_primes:
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MR_BASIS:
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

# --- Pollard-Brent ---
def pollard_brent(n, rng):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = rng.randrange(2, n - 1)
        c = rng.randrange(1, n - 1)
        m = rng.randrange(1, 100)
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
                g = _gcd(q, n)
                k += m
            r *= 2
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = _gcd(abs(x - ys), n)
                if g > 1:
                    break
        if g != n:
            return g

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# --- sieve of small primes ---
_SIEVE_LIMIT = 5000
def _sieve(n):
    bs = bytearray([1]) * (n + 1)
    bs[0:2] = b'\0\0'
    for i in range(2, int(n ** 0.5) + 1):
        if bs[i]:
            bs[i * i::i] = b'\0' * len(bs[i * i::i])
    return [i for i in range(2, n + 1) if bs[i]]

_SMALL_PRIMES = _sieve(_SIEVE_LIMIT)

# primes <= 1000 for fast stripping
_SMALL_STRIP = [p for p in _SMALL_PRIMES if p <= 1000]

_rng = random.Random(12345)

def is_psmooth(n, p):
    """Return True if every prime factor of n is <= p."""
    m = n
    for sp in _SMALL_STRIP:
        if sp > p:
            break
        if sp * sp > m:
            break
        if m % sp == 0:
            while m % sp == 0:
                m //= sp
    if m == 1:
        return True
    if m > p:
        # m has no prime factor <= min(p,1000). If m itself is prime and <= p, ok;
        # if m is composite, it has a factor > 1000; need to check.
        if m <= p:
            # m could be prime or composite with factors <= p (all >1000)
            # if m is prime, it's <= p -> ok. If composite, factors could be > p.
            # Check: is m prime?
            if is_prime(m):
                return True
            # composite: factor it
            f = pollard_brent(m, _rng)
            if f > p or m // f > p:
                return False
            # both factors <= p, but need full check
            return is_psmooth(f, p) and is_psmooth(m // f, p)
        else:
            # m > p, and no small factor. If m is prime, m > p -> fail.
            # If composite, it has a prime factor; could be <= p.
            if is_prime(m):
                return False  # m prime and m > p
            f = pollard_brent(m, _rng)
            if f > p or m // f > p:
                return False
            return is_psmooth(f, p) and is_psmooth(m // f, p)
    else:
        # m <= p, no small factor. If prime, ok. If composite, factors could be > p? No: m<=p so any factor <= m <= p.
        return True

def construct():
    global _rng
    rng = random.Random(98765)
    _rng = rng

    best_k = 0
    best_p = None

    # Candidate primes to test. We want p large enough that p^2+j tends to be p-smooth.
    # Empirically, medium primes give longer runs. Search a range of primes.
    # Generate primes up to some bound, then test them.
    # We'll test primes in increasing order of "promise" but must be time-bounded.
    # Strategy: test primes from a set, for each count consecutive psmooth values.

    # Sieve primes up to 200000 for candidates (p up to 200k)
    # Also consider larger primes via random sampling? Keep it simple: sieve up to 200k.
    P_MAX = 200000
    primes = _sieve(P_MAX)

    # For each prime p, count k = number of consecutive n = p^2+1, p^2+2, ... that are p-smooth.
    # Stop when one fails or k reaches 200 or time runs out.
    # To maximize total work within deadline, test primes in a good order.
    # Larger p -> p^2 is bigger -> harder to be p-smooth (relatively), but the density of p-smooth
    # numbers near p^2 is roughly psi(p^2, p)/p^2. For p^2 with bound p, this is related to Dickman.
    # Empirically medium p (1000-50000) may give best runs. Test in that range first, then expand.

    # Order: test primes from 1000 upward, but interleave to find a good one fast.
    # We'll just iterate primes and test each, keeping best.
    # To use time well, for each p we can test up to 200 values but stop early on first failure.

    # Precompute nothing; just loop.
    # Sort primes so we test promising ones first. Start from smaller to larger? 
    # Actually, let's test a spread: first pass over primes in [500, 20000], then expand.
    # Simpler: iterate all primes in [100, P_MAX] in ascending order, but that wastes time on small p.
    # Better: iterate in descending order from a high bound down? No, large p are harder.
    # Let's do: for p in primes, but skip very small p (< 100) and very large p where p^2 > 10^18 (not an issue here).
    # We'll test in order of increasing p but start at p=100.

    # Actually, the best k is likely found at moderate p. Let's test p in a range and also try to
    # be clever: for each p, the run length is random-ish. We just need to find the max.
    # With 120s, we can test maybe 10^4-10^5 primes with ~10-50 checks each.
    # Let's iterate primes from 100 to P_MAX, testing each, and keep the best.

    # To be safe on time, we break the outer loop when time runs out.
    for p in primes:
        if not _time_ok():
            break
        if p < 100:
            continue
        if p > P_MAX:
            break
        k = 0
        base = p * p
        while k < 200:
            if not _time_ok():
                break
            n = base + k + 1
            if is_psmooth(n, p):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p

    # If we didn't find anything, fall back to a trivial valid answer.
    if best_p is None:
        # Find any prime p and any k>=1. Use p=2, n=5 (2^2+1=5, 5>2, not 2-smooth). 
        # p=3: 3^2+1=10, factors 2,5; 5>3, not 3-smooth. 
        # p=5: 25+1=26=2*13, 13>5. 
        # We need a valid (p,k). Let's just return p=3, k=0? No, k>=1 required.
        # Actually the task says 1<=k<=200 and every integer p^2+1..p^2+k must be p-smooth.
        # If no such p found in our search, we need a fallback. But we should find some.
        # p=2: 4+1=5 not 2-smooth. p=3: 10 not 3-smooth (5>3). p=5: 26 not 5-smooth.
        # p=7: 49+1=50=2^2*5, 5<=7 ok; 49+2=51=3*17, 17>7 fail. So k=1 for p=7.
        # Let's verify p=7, k=1: 50 is 7-smooth (factors 2,5 both <=7). Yes.
        best_p = 7
        best_k = 1

    return {'p': best_p, 'k': best_k}
