import time
import math
import random
import sys

# ---------- Deterministic Miller-Rabin for 64-bit ----------
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def _is_prime_mr(n):
    if n < 2:
        return False
    for p in _MR_BASES:
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MR_BASES:
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

# ---------- Pollard-Brent ----------
def _pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
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

def _factor(n, result):
    if n <= 1:
        return
    if _is_prime_mr(n):
        result.append(n)
        return
    d = _pollard_brent(n)
    _factor(d, result)
    _factor(n // d, result)

# ---------- Sieve ----------
def _sieve(limit):
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(limit + 1) if is_prime[i]]

def _is_p_smooth_trial(n, p, small_primes):
    """Check if n is p-smooth using trial division by small_primes up to p."""
    t = n
    for q in small_primes:
        if q > p:
            break
        if q * q > t:
            break
        while t % q == 0:
            t //= q
    if t == 1:
        return True
    # t is either prime or composite with all factors > smallest prime not tried
    if _is_prime_mr(t):
        return t <= p
    # t is composite; factor it
    factors = []
    _factor(t, factors)
    return all(f <= p for f in factors)

def _is_p_smooth_hybrid(n, p, small_primes, small_limit):
    """Check if n is p-smooth: trial divide by small_primes, then Pollard-Brent on remainder."""
    t = n
    for q in small_primes:
        if q * q > t:
            break
        while t % q == 0:
            t //= q
    if t == 1:
        return True
    # Now t has no factors <= small_limit
    # If t is prime, check if t <= p
    if _is_prime_mr(t):
        return t <= p
    # t is composite with all prime factors > small_limit
    # Factor t and check all factors <= p
    factors = []
    _factor(t, factors)
    return all(f <= p for f in factors)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin

    # Precompute small primes up to 100000 for fast trial division
    SIEVE_LIMIT = 100000
    small_primes = _sieve(SIEVE_LIMIT)
    small_prime_set = set(small_primes)

    best_k = 0
    best_p = 3

    # Phase 1: Search small primes (up to 100000) with full trial division
    # This is fast and might find a good local maximum early.
    for p in small_primes:
        if time.time() > deadline:
            break
        p_sq = p * p
        k = 0
        for i in range(1, 201):
            if time.time() > deadline:
                break
            val = p_sq + i
            if _is_p_smooth_trial(val, p, small_primes):
                k = i
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
            if best_k >= 200:
                return {'p': best_p, 'k': best_k}

    # Phase 2: CRT-based construction
    # We want to find p such that for i=1..k, p^2+i is p-smooth.
    # We force p^2+i to have a factor q_i > p by setting p^2 ≡ -i (mod q_i).
    # We must avoid p^2+i being divisible by any prime q <= p (unless the remainder is 1).
    
    random.seed(42)
    
    # Primes available for "forbidden" factors (q > p)
    # We need q > p. If we search p up to 10^9, we need q up to 10^9.
    # We can generate random large primes or use a precomputed list if we had one.
    # Generating primes on the fly is expensive. 
    # Let's precompute primes up to 10^6 for the "small primes" to avoid.
    # For "large primes" q_i, we can pick random candidates and test primality.
    
    # To make CRT efficient, we should use small moduli q_i if possible, but q_i MUST be > p.
    # This implies p must be small if we want small q_i.
    # If p is large (e.g. 10^6), q_i must be > 10^6.
    
    # Strategy:
    # 1. Pick a target k (e.g. 10, 20, 50).
    # 2. Pick k distinct primes q_1, ..., q_k.
    # 3. Solve x^2 ≡ -i (mod q_i) for x. This requires -i to be a quadratic residue mod q_i.
    #    By Legendre symbol, (-i/q_i) = 1.
    #    We can filter q_i to satisfy this.
    # 4. Combine with CRT to get p mod M = prod(q_i).
    # 5. Find a prime p = r + j*M that is > max(q_i) and avoids "small prime" collisions.
    
    # Let's iterate over k from high to low, trying to construct a valid p.
    # We'll start with small k and increase, or start with large k and decrease?
    # Larger k is harder. Let's try to maximize k.
    # We'll try k = 20, 30, 40... and see if we can find a p.
    
    # To avoid "small prime" collisions:
    # We need p^2 + i NOT divisible by any prime q <= p.
    # If p^2 + i is divisible by q <= p, then p^2 ≡ -i (mod q).
    # We can add these as "forbidden" congruences in a secondary CRT or just check.
    # Checking 200 values against ~10^6 primes is too slow if done naively for every candidate p.
    # However, if we find a candidate p, we just check p^2+1..p^2+k for smoothness.
    # The CRT ensures p^2+i has a large factor q_i > p.
    # If p^2+i is also divisible by a small q, it might still be p-smooth (if the rest is small).
    # But if p^2+i = q_i * (small part), and small part is p-smooth, we are good.
    # The main risk is p^2+i being prime (impossible since q_i > 1 divides it) or having a prime factor > p other than q_i.
    # If p^2+i = q_i * r, and r is composite, r might have a factor > p.
    # So we still need to verify smoothness of r.
    
    # This approach doesn't *guarantee* p-smoothness, it just *increases the probability* by ensuring a large factor exists.
    # Wait, if q_i > p, then p^2+i is NOT p-smooth UNLESS the remaining factor is 1.
    # Ah, the condition is "all prime factors at most p".
    # If p^2+i has a factor q_i > p, then it FAILS the condition.
    # So my CRT logic was INVERTED.
    # I want p^2+i to be p-smooth.
    # This means p^2+i must NOT have any prime factor > p.
    # So I should AVOID having factors > p.
    # This is the opposite of what I just derived.
    
    # Correct Logic:
    # We want p^2+i to be p-smooth.
    # This implies p^2+i < p^2 + k.
    # Any prime factor of p^2+i must be <= p.
    # So p^2+i must be a product of primes <= p.
    # This is a very restrictive condition.
    
    # Let's go back to the search strategy.
    # The current program searches primes p and checks k.
    # The "stagnation" suggests we need a better search.
    
    # What if we search for p such that p^2+i is SMALL?
    # No, p^2 is fixed.
    
    # Let's try a different structural change:
    # Instead of scanning all primes, we can use the property that for large p, p^2+i is likely to have a large prime factor.
    # We want p^2+i to be "smooth".
    # Smooth numbers are rare.
    # The probability that a number N is y-smooth is roughly rho(log N / log y).
    # Here N ≈ p^2, y = p.
    # log N ≈ 2 log p, log y = log p.
    # Ratio ≈ 2.
    # rho(2) ≈ 0.3.
    # So ~30% of numbers are p-smooth?
    # But we need a RUN of k such numbers.
    # Probability ≈ 0.3^k.
    # For k=10, 0.3^10 ≈ 0.000059.
    # So we need to check ~17000 primes to find one run of 10.
    # For k=20, 0.3^20 ≈ 3e-11. Need 3e10 primes. Impossible.
    
    # So finding long runs is hard.
    # The seed program found k=8 (score 2.6667 implies k=8 if seed=3? No, score is value/seed. If seed k=3, and value=8, score=2.66. If seed k=6, value=16?
    # The prompt says "score 2.6667".
    # If the seed program had k=3, and we found k=8, score is 8/3 = 2.66.
    # If we find k=10, score is 10/3 = 3.33.
    # If we find k=12, score is 4.0.
    
    # The bottleneck is checking smoothness.
    # Trial division up to p is O(p).
    # If p=10^6, checking 200 numbers takes 200 * 10^6 / log(10^6) operations? No, trial division by primes up to p.
    # Number of primes up to 10^6 is ~78,000.
    # 200 * 78,000 = 15.6 million ops. In Python, this is ~1-2 seconds.
    # So we can check maybe 50 primes per second for p=10^6.
    # In 100 seconds, we can check 5000 primes.
    # Expected hits for k=10: 5000 * 0.000059 ≈ 0.3. Unlikely.
    
    # We need to check LARGER p?
    # For larger p, the probability rho(2) is constant, but the cost of checking increases.
    # Wait, rho(2) is the probability that a number is 2-smooth in the log-log sense?
    # Dickman function rho(u) where u = log N / log y.
    # Here u = 2.
    # rho(2) = 1 - log 2 ≈ 0.306.
    # So the probability is constant.
    # But the cost of verification increases with p.
    # So smaller p is better for search speed.
    
    # Why did the seed program only search up to p=2000?
    # "The current program only searches up to p=2000 with naive trial division."
    # If we optimize the trial division, we can search much further.
    
    # Let's implement a highly optimized search for small p (up to 10^5 or 10^6).
    # Use a sieve to precompute primes.
    # Use vectorized operations if possible? No, standard Python.
    # But we can speed up the smoothness check.
    
    # Optimization:
    # 1. Precompute primes up to 10^6.
    # 2. For a given p, check p^2+1, ..., p^2+k.
    # 3. Instead of full factorization, just trial divide by primes <= p.
    # 4. If the remainder is 1, it's smooth.
    # 5. If the remainder is prime and <= p, it's smooth.
    # 6. If the remainder is composite, we need to check if its factors are <= p.
    #    Since we divided by all primes <= p, any remaining factor must be > p.
    #    So if remainder > 1, it FAILS.
    #    WAIT.
    #    If we trial divide by ALL primes <= p, and the remainder is > 1, then the remainder has a prime factor > p.
    #    So the number is NOT p-smooth.
    #    So the check is simply:
    #    Divide n by all primes q <= p.
    #    If remainder is 1, it is p-smooth.
    #    Otherwise, it is not.
    
    # This is MUCH faster!
    # We don't need Pollard-Brent or Miller-Rabin for the check itself, only for finding primes p.
    # And for trial division, we stop when q*q > t or q > p.
    # Actually, we can just iterate through primes <= p and divide.
    
    # Let's rewrite the check:
    def is_p_smooth_fast(n, p, primes):
        t = n
        for q in primes:
            if q > p:
                break
            if q * q > t:
                break
            while t % q == 0:
                t //= q
        return t == 1

    # Now, how many primes can we check?
    # For p=10^5, pi(p) ≈ 9592.
    # Checking 200 numbers: 200 * 9592 ≈ 2 million divisions.
    # In Python, ~0.1s per prime.
    # 100 seconds -> 1000 primes.
    # Expected hits for k=10: 1000 * 0.3^10 ≈ 0.06. Still low.
    
    # We need to go to smaller p?
    # For p=1000, pi(p) = 168.
    # 200 * 168 = 33,600 divisions. ~0.001s.
    # 100 seconds -> 100,000 primes.
    # But there are only 168 primes up to 1000.
    # So we check all primes up to 1000 in 0.1s.
    # Then go to 10^4. pi(10^4) = 1229.
    # 200 * 1229 = 245,800 divisions. ~0.02s per prime.
    # 100 seconds -> 5000 primes.
    # Total primes checked: 168 + 5000 ≈ 5000.
    # Expected hits for k=10: 5000 * 0.000059 ≈ 0.3.
    
    # It seems k=10 is hard to find by random search.
    # Maybe the seed program found k=8 with p < 100?
    # If p is small, the "smooth" condition is easier to satisfy?
    # No, rho(2) is constant.
    
    # Let's just implement the fast trial division and search as many primes as possible.
    # We will search primes in increasing order.
    
    # Precompute primes up to 10^6.
    primes = _sieve(1000000)
    
    # Iterate through primes
    # We need to be careful with time.
    
    # Let's run this.
    
    for p in primes:
        if time.time() > deadline:
            break
        if p > 10**6:
            break
        p_sq = p * p
        k = 0
        for i in range(1, 201):
            val = p_sq + i
            if is_p_smooth_fast(val, p, primes):
                k = i
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
            if best_k >= 200:
                return {'p': best_p, 'k': best_k}
            
    return {'p': best_p, 'k': best_k}
