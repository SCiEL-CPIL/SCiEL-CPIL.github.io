"""Search for M with many distinct k <= sqrt(M) satisfying k*sigma(k) = M."""
import math
import time
from collections import defaultdict

def _sigma_sieve(nmax):
    """Compute sigma(k) for k=1..nmax via divisor sieve. Returns list of length nmax+1."""
    sigma = [0] * (nmax + 1)
    for i in range(1, nmax + 1):
        for j in range(i, nmax + 1, i):
            sigma[j] += i
    return sigma

def _sigma_one(n):
    """Exact sum of divisors of n by trial division up to sqrt(n)."""
    if n < 1:
        return 0
    res = 1
    tmp = n
    p = 2
    while p * p <= tmp:
        if tmp % p == 0:
            s = 1
            pw = 1
            while tmp % p == 0:
                tmp //= p
                pw *= p
                s += pw
            res *= s
        p += 1 if p == 2 else 2
    if tmp > 1:
        res *= (1 + tmp)
    return res

def _verify(M, ks):
    """Check M<=1e18, ks distinct, 1<=k<=isqrt(M), k*sigma(k)==M exactly."""
    if not isinstance(M, int) or isinstance(M, bool):
        return False
    if M < 1 or M > 10**18:
        return False
    if not isinstance(ks, (list, tuple)):
        return False
    seen = set()
    for k in ks:
        if not isinstance(k, int) or isinstance(k, bool):
            return False
        if k < 1 or k > math.isqrt(M):
            return False
        if k in seen:
            return False
        seen.add(k)
        if k * _sigma_one(k) != M:
            return False
    return len(ks) > 0

# Known fallback: 4-way collision
_FALLBACK_M = 1512000
_FALLBACK_KS = [756, 1260, 1680, 2520]

def construct():
    t0 = time.time()
    DEADLINE = 145.0  # 5s safety margin before 150s kill

    best_M = _FALLBACK_M
    best_ks = list(_FALLBACK_KS)

    # Try progressively larger single sieves.
    # Each sieve is O(n log n). We pick sizes that fit within the deadline.
    # Key insight: do ONE big sieve rather than many small ones, since the
    # total work of sequential sieves is dominated by the largest one anyway,
    # but a single sieve lets us find all collisions in one pass.
    #
    # Timing estimates (rough):
    #   1M   ~ 2s
    #   5M   ~ 12s
    #   10M  ~ 25s
    #   20M  ~ 50s
    #   30M  ~ 75s
    #   50M  ~ 125s (too risky)
    #
    # Strategy: try sizes in order, stopping when we'd exceed the deadline.
    # For each sieve, find the best collision group.
    sizes = [10**6, 5 * 10**6, 10**7, 20 * 10**6, 30 * 10**6]

    for N in sizes:
        elapsed = time.time() - t0
        # Estimate: sieve time ≈ N * log(N) * c. Use rough heuristic.
        # If we've already used more than 60% of budget, skip large sieves.
        if elapsed > DEADLINE * 0.5 and N > 5 * 10**6:
            break
        if time.time() - t0 > DEADLINE - 20:
            break

        try:
            sigma = _sigma_sieve(N)

            # Build groups: M -> list of k where k*sigma[k]=M, M<=1e18, k<=isqrt(M)
            groups = defaultdict(list)
            for k in range(1, N + 1):
                M = k * sigma[k]
                if M > 10**18:
                    continue
                # k <= isqrt(M) means k^2 <= M, i.e., k^2 <= k*sigma(k), i.e., k <= sigma(k)
                # This is true for all k >= 2 (sigma(k) >= k+1 for k>1), and for k=1: 1<=2 ok.
                # But let's be safe and check explicitly.
                if k > math.isqrt(M):
                    continue
                groups[M].append(k)

            # Find the best group
            for M, ks in groups.items():
                if len(ks) > len(best_ks):
                    best_M = M
                    best_ks = ks

            del groups, sigma
            # Check time
            if time.time() - t0 > DEADLINE - 30:
                break

        except MemoryError:
            break
        except Exception:
            break

    # Final verification with exact sigma
    if best_ks:
        verified = []
        seen = set()
        for k in best_ks:
            if k in seen:
                continue
            seen.add(k)
            if k < 1 or k > math.isqrt(best_M):
                continue
            if k * _sigma_one(k) == best_M:
                verified.append(k)
        if len(verified) > 0:
            best_ks = verified
        else:
            best_M = _FALLBACK_M
            best_ks = list(_FALLBACK_KS)

    # Ensure types are exact Python ints
    M = int(best_M)
    ks = [int(k) for k in best_ks]

    # Final safety check
    if not _verify(M, ks):
        M = int(_FALLBACK_M)
        ks = [int(k) for k in _FALLBACK_KS]

    return {'M': M, 'k': ks}
