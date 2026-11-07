import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin for 150s wall-clock

    # ---------- exact sigma (trial division, good enough for k <= ~1e9) ----------
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
            result *= (temp + 1)
        return result

    def _fallback():
        ks = [336, 372, 434]
        M = 333312
        if all(k * sigma_exact(k) == M for k in ks):
            return {'M': int(M), 'k': [int(x) for x in ks]}
        return {'M': 1, 'k': [1]}

    # ============================================================
    # PASS A: multiplicative-structure enumeration.
    # Enumerate k = 2^a 3^b 5^c 7^d 11^e with bounded exponents.
    # sigma(k) = prod (p^{a+1}-1)/(p-1) computed exactly.
    # M = k*sigma(k). Group by M.  Small primes keep M ~ k^2
    # so k <= isqrt(M) is automatic, and M <= 10^18 is enforced.
    # ============================================================
    best_count = 0
    best_M = 0
    best_ks = []

    try:
        primes = [2, 3, 5, 7, 11]
        exps = [11, 8, 6, 5, 4]
        caps = [2**11, 3**8, 5**6, 7**5, 11**4]

        m_to_ks = defaultdict(list)

        def rec(idx, k, s):
            nonlocal best_count, best_M, best_ks
            if time.time() > deadline - 90:
                return
            if idx == len(primes):
                M = k * s
                if M <= 10**18 and 1 <= k <= math.isqrt(M):
                    m_to_ks[M].append(k)
                    c = len(m_to_ks[M])
                    if c > best_count:
                        best_count = c
                        best_M = M
                        best_ks = list(m_to_ks[M])
                return
            p = primes[idx]
            cap = caps[idx]
            pk = 1
            ps = 1
            for _ in range(exps[idx] + 1):
                rec(idx + 1, k * pk, s * ps)
                if pk > cap // p:
                    break
                pk *= p
                ps = ps * p + 1

        rec(0, 1, 1)
    except Exception:
        pass

    # ============================================================
    # PASS B: broad divisor-sum sieve (numpy vectorized).
    # Catches collisions across the full k range [1, N_MAX].
    # ============================================================
    b_count = 0
    b_M = 0
    b_ks = []
    try:
        import numpy as np
        N_MAX = 10_000_000
        sigma_arr = np.zeros(N_MAX + 1, dtype=np.uint64)
        for d in range(1, N_MAX + 1):
            if time.time() > deadline - 35:
                break
            sigma_arr[d::d] += d

        m2_first = {}
        m2_ks = {}
        for k in range(1, N_MAX + 1):
            if time.time() > deadline - 35:
                break
            M = k * int(sigma_arr[k])
            if M > 10**18 or k > math.isqrt(M):
                continue
            if M in m2_ks:
                m2_ks[M].append(k)
                c = len(m2_ks[M])
                if c > b_count:
                    b_count = c
                    b_M = M
                    b_ks = m2_ks[M]
            elif M in m2_first:
                m2_ks[M] = [m2_first.pop(M), k]
                if 2 > b_count:
                    b_count = 2
                    b_M = M
                    b_ks = m2_ks[M]
            else:
                m2_first[M] = k
    except Exception:
        pass

    # ---------- Choose the better of the two passes ----------
    if b_count > best_count:
        best_count, best_M, best_ks = b_count, b_M, b_ks

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
