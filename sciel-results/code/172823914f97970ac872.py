import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin for the 150s wall-clock limit

    # ---------- Exact sigma via trial division (used only for verification) ----------
    def sigma_exact(n):
        if n == 1:
            return 1
        result = 1
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                p_power = 1
                while temp % p == 0:
                    temp //= p
                    p_power *= p
                result *= (p_power * p - 1) // (p - 1)
            p += 1 if p == 2 else 2
        if temp > 1:
            result *= (temp * temp - 1) // (temp - 1)
        return result

    def _fallback():
        ks = [336, 372, 434]
        M = 333312
        if all(k * sigma_exact(k) == M for k in ks):
            return {'M': int(M), 'k': [int(x) for x in ks]}
        return {'M': 1, 'k': [1]}

    best_count = 0
    best_M = 0
    best_ks = []

    # ============================================================
    # PASS A: Broad sieve over k up to a large bound.
    # Compute sigma(k) for all k in [1, N] using a divisor-sum sieve.
    # Group by M = k*sigma(k) and track the best collision group.
    # ============================================================
    try:
        import numpy as np
        # Choose N so that k*sigma(k) can reach ~10^18 for many k.
        # sigma(k) is typically ~ k * log(log(k)), so k ~ sqrt(M/log(log(k))) ~ 10^9 / small factor.
        # But we need memory for sigma array. Let's use N = 50_000_000 (50M) which fits in ~400MB as uint64.
        # Actually, let's be more careful. We need N such that there are many collisions.
        # The current program uses 10M. Let's try 30M first to see if it fits in time.
        
        N = 30_000_000
        # Check if we have enough time. Sieve O(N log N) ~ 30M * 17 ~ 500M ops. In numpy this is fast.
        sigma_arr = np.zeros(N + 1, dtype=np.uint64)
        
        for d in range(1, N + 1):
            if time.time() > deadline - 40:
                break
            # Vectorized addition
            sigma_arr[d::d] += d
        
        # Now group by M
        m_to_ks = defaultdict(list)
        for k in range(1, N + 1):
            if time.time() > deadline - 40:
                break
            M = k * int(sigma_arr[k])
            if M > 10**18:
                continue
            if k > math.isqrt(M):
                continue
            m_to_ks[M].append(k)
            c = len(m_to_ks[M])
            if c > best_count:
                best_count = c
                best_M = M
                best_ks = m_to_ks[M][:]  # copy to avoid reference issues
    except Exception:
        pass

    # ============================================================
    # PASS B: Targeted search over smooth numbers (small primes).
    # This catches cases where k is very smooth and M is highly composite.
    # We enumerate k = 2^a 3^b 5^c 7^d 11^e 13^f with bounded exponents.
    # ============================================================
    try:
        primes = [2, 3, 5, 7, 11, 13]
        # Max exponents such that k <= 10^9 (so M ~ k*sigma(k) <= 10^18 is plausible)
        # 2^30 ~ 10^9, 3^18 ~ 3.8e8, 5^13 ~ 1.2e9, 7^11 ~ 1.9e9, 11^9 ~ 2.3e9, 13^8 ~ 8e8
        # Let's use slightly smaller to keep the search space manageable.
        exps = [29, 17, 12, 10, 8, 7]
        
        m_to_ks_b = defaultdict(list)
        
        def rec(idx, k, s):
            nonlocal best_count, best_M, best_ks
            if time.time() > deadline - 20:
                return
            if idx == len(primes):
                M = k * s
                if M <= 10**18 and 1 <= k <= math.isqrt(M):
                    m_to_ks_b[M].append(k)
                    c = len(m_to_ks_b[M])
                    if c > best_count:
                        best_count = c
                        best_M = M
                        best_ks = m_to_ks_b[M][:]
                return
            p = primes[idx]
            pk = 1
            ps = 1
            for _ in range(exps[idx] + 1):
                rec(idx + 1, k * pk, s * ps)
                pk *= p
                ps = ps * p + 1

        rec(0, 1, 1)
    except Exception:
        pass

    # ---------- Exact verification of the chosen list ----------
    final_ks = []
    if best_M > 0:
        for k in best_ks:
            if isinstance(k, int) and 1 <= k <= math.isqrt(best_M):
                if k * sigma_exact(k) == best_M:
                    final_ks.append(k)

    if not final_ks:
        return _fallback()

    return {'M': int(best_M), 'k': [int(x) for x in final_ks]}
