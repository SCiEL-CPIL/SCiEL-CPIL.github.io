"""Rewritten: targeted sigma-fiber search near a/b = 3/2."""
import time
from fractions import Fraction
from math import isqrt

MAX_N = 10_000_000
DEADLINE = 85.0  # safety margin under 90s kill

def _build_spf_sigma(limit):
    """Return (spf, sigma) arrays for 1..limit using linear sieve."""
    spf = [0] * (limit + 1)
    sigma = [0] * (limit + 1)
    sigma[0] = 0
    sigma[1] = 1
    primes = []
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, limit + 1):
        if is_prime[i]:
            primes.append(i)
            spf[i] = i
            sigma[i] = 1 + i
        for p in primes:
            v = i * p
            if v > limit:
                break
            is_prime[v] = False
            spf[v] = p
            if p == spf[i]:
                # p divides i, so p^2 divides v
                # find p-power part of i
                pk = 1
                ppow = 1
                m = i
                while m % p == 0:
                    ppow *= p
                    m //= p
                pk = ppow
                # sigma(v) = sigma(i/p^e) * (1 + p + ... + p^{e+1})
                # where i = m * p^e
                rest = m
                # sum_{j=0}^{e+1} p^j = (p^{e+2}-1)/(p-1)
                s_new = 1
                term = 1
                for _ in range(e + 2) if False else range(0):
                    pass
                # compute directly
                s = 1
                t = 1
                for _ in range(e + 2):
                    s += t
                    t *= p
                # but we need sigma(i) = sigma(m) * (1+...+p^e)
                # sigma(m) we don't have directly; recompute via spf is heavy.
                # Instead use recurrence: sigma(v) = sigma(i) * (1 + p^{e+1}) / (1 + p^e)? No.
                break
    # The linear sieve sigma recurrence is tricky; use classic sieve instead.
    return None, None

def _sigma_sieve(limit):
    """Classic divisor-sum sieve: sigma[n] = sum of divisors of n."""
    sigma = [0] * (limit + 1)
    for d in range(1, limit + 1):
        for m in range(d, limit + 1, d):
            sigma[m] += d
    return sigma

def _sigma_via_spf(limit):
    """Compute sigma using smallest-prime-factor recurrence (fast)."""
    spf = [0] * (limit + 1)
    primes = []
    for i in range(2, limit + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
        for p in primes:
            if p > spf[i] or i * p > limit:
                break
            spf[i * p] = p
    sigma = [0] * (limit + 1)
    sigma[1] = 1
    for n in range(2, limit + 1):
        p = spf[n]
        m = n // p
        if m % p == 0:
            # p divides m
            # find p^e || n
            pk = 1
            mm = m
            while mm % p == 0:
                pk *= p
                mm //= p
            # n = mm * p * pk = mm * p^{e+1}, where m = mm * pk, e = v_p(m)
            # sigma(n) = sigma(mm) * (1 + p + ... + p^{e+1})
            # sigma(m) = sigma(mm) * (1 + ... + p^e)
            # We need sigma(mm). Recompute: sigma(mm) = sigma(m) / (1+...+p^e)
            # But integer division risky; instead track p-power sum.
            # Let S_e = 1 + p + ... + p^e. sigma(m) = sigma(mm)*S_e.
            # sigma(n) = sigma(mm)*S_{e+1} = sigma(m) * S_{e+1}/S_e.
            # S_{e+1}/S_e is not integer in general. Use sigma(mm) directly.
            # Compute sigma(mm) by dividing sigma[m] by S_e (exact).
            S_e = 1
            t = 1
            for _ in range(e + 1) if False else range(0):
                pass
            # recompute S_e
            S_e = 1
            t = 1
            cnt = 0
            mm2 = m
            while mm2 % p == 0:
                t *= p
                S_e += t
                mm2 //= p
                cnt += 1
            # S_e = 1 + p + ... + p^{cnt}
            sig_mm = sigma[m] // S_e
            # S_{cnt+1} = S_e + p^{cnt+1}
            t2 = 1
            for _ in range(cnt + 1):
                t2 *= p
            S_next = S_e + t2
            sigma[n] = sig_mm * S_next
        else:
            # p does not divide m
            sigma[n] = sigma[m] * (1 + p)
    return sigma

def _best_pair(sigma, max_n, deadline):
    """Find best (a,b) with a,b<=max_n, sigma(a)=sigma(b), a/b near 3/2, a != b*3/2 exactly."""
    target = Fraction(3, 2)
    best = None
    best_err = None
    t0 = time.time()
    # Iterate over b; a should be near 1.5*b
    # For each b, consider a in a small window around 3b/2
    # To be thorough but fast, step through b and check a few nearby a
    # Use the structure: a = 3k, b = 2k gives ratio exactly 3/2 -> invalid (err 0)
    # So we need a/b != 3/2 exactly but close.
    # Strategy: for each b, candidate a values near 3b/2: a = floor(3b/2)+d for small d
    # Check sigma[a] == sigma[b]
    # To keep within deadline, limit b range and window
    max_b = min(max_n, 2_000_000)  # b up to 2e6, a up to 3e6
    # Actually a <= max_n so b <= max_n, and a ~ 1.5 b <= max_n => b <= max_n*2/3
    max_b = min(max_n, (max_n * 2) // 3)
    window = 40
    for b in range(2, max_b + 1):
        if time.time() - t0 > deadline:
            break
        sb = sigma[b]
        base = (3 * b) // 2
        # a must satisfy a/b near 3/2 => a near 3b/2
        # a in [base - window, base + window]
        lo = max(1, base - window)
        hi = min(max_n, base + window)
        # Also ensure a/b != 3/2 exactly: 2a != 3b
        for a in range(lo, hi + 1):
            if 2 * a == 3 * b:
                continue  # exact 3/2 -> invalid (err 0)
            if sigma[a] != sb:
                continue
            # compute |a/b - 3/2| = |2a - 3b| / (2b)
            num = abs(2 * a - 3 * b)
            den = 2 * b
            err = Fraction(num, den)
            if best_err is None or err < best_err:
                best_err = err
                best = (a, b)
    return best, best_err

def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE
    # Build sigma up to MAX_N
    sigma = _sigma_via_spf(MAX_N)
    # Verify a few
    # sigma[1]=1, sigma[2]=3, sigma[3]=4, sigma[4]=7, sigma[6]=12
    # Find best pair
    best, best_err = _best_pair(sigma, MAX_N, deadline - (time.time() - t_start) - 2.0)
    if best is None:
        # fallback: known valid pair? Need sigma(a)=sigma(b), a!=b, a/b near 3/2
        # sigma(1)=1; sigma(2)=3; sigma(3)=4; sigma(4)=7; sigma(5)=6; sigma(6)=12
        # sigma(7)=8; sigma(8)=15; sigma(9)=13; sigma(10)=18; sigma(11)=12; sigma(12)=28
        # sigma(13)=14; sigma(14)=24; sigma(15)=24; sigma(16)=31; sigma(17)=18
        # sigma(6)=12, sigma(11)=12 -> a=6,b=11 ratio 6/11~0.545; a=11,b=6 ratio 11/6~1.833
        # |11/6 - 3/2| = |11/6 - 9/6| = 2/6 = 1/3
        # sigma(14)=24, sigma(15)=24 -> 15/14 ~1.071, |15/14-3/2|=|15/14-21/14|=6/14=3/7
        # sigma(10)=18, sigma(17)=18 -> 17/10=1.7, |17/10-3/2|=|17/10-15/10|=2/10=1/5
        best = (17, 10)
        best_err = Fraction(1, 5)
    a, b = best
    # Ensure types are Python int
    a = int(a)
    b = int(b)
    return [a, b]
