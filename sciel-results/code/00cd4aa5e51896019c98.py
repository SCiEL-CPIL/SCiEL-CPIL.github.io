import time
import math
from collections import defaultdict


def construct():
    start = time.time()
    deadline = start + 110.0  # Safety margin before 150s kill

    # Fallback known solution: 336*sigma(336) = 372*sigma(372) = 434*sigma(434) = 333312
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3

    # ---------- helpers ----------
    def factorize(n):
        """Factorize n via trial division. Returns list of (p, exp)."""
        if n == 1:
            return []
        factors = []
        d = 2
        while d * d <= n:
            if n % d == 0:
                cnt = 0
                while n % d == 0:
                    n //= d
                    cnt += 1
                factors.append((d, cnt))
            d += 1 if d == 2 else 2
        if n > 1:
            factors.append((n, 1))
        return factors

    def sigma_from_factors(factors):
        """Compute sigma from factor list [(p, exp), ...]."""
        result = 1
        for p, e in factors:
            result *= (p ** (e + 1) - 1) // (p - 1)
        return result

    def sigma_exact(n):
        if n == 1:
            return 1
        return sigma_from_factors(factorize(n))

    def gen_divisors(factors):
        """Generate all divisors from factor list."""
        divs = [1]
        for p, e in factors:
            new_divs = []
            for d in divs:
                pe = 1
                for _ in range(e + 1):
                    new_divs.append(d * pe)
                    pe *= p
            divs = new_divs
        return divs

    # ---------- Phase 1: Sieve sigma up to N, group by M ----------
    N = 5_000_000
    if time.time() < deadline - 40:
        sigma = [0] * (N + 1)
        for i in range(1, N + 1):
            for j in range(i, N + 1, i):
                sigma[j] += i

        m_to_k = defaultdict(list)
        for k in range(2, N + 1):
            M = k * sigma[k]
            if M <= 10**18:
                m_to_k[M].append(k)

        for M, ks in m_to_k.items():
            if len(ks) > best_count:
                best_count = len(ks)
                best_M = M
                best_k = ks[:]

        del sigma
        del m_to_k

    # ---------- Phase 2: Polish pass on current best_M via divisor enumeration ----------
    if time.time() < deadline - 10 and best_M is not None:
        try:
            factors_M = factorize(best_M)
            divisors = gen_divisors(factors_M)
            isqrt_M = math.isqrt(best_M)
            valid_k = []
            for k in divisors:
                if 1 < k <= isqrt_M:
                    try:
                        if k * sigma_exact(k) == best_M:
                            valid_k.append(k)
                    except Exception:
                        continue
            if len(valid_k) > best_count:
                best_count = len(valid_k)
                best_k = valid_k
        except Exception:
            pass

    # ---------- Phase 3: Meet-in-the-middle using multiplicative structure ----------
    # Decompose k = a * b where gcd(a,b)=1, sigma(k) = sigma(a)*sigma(b), M = a*sigma(a)*b*sigma(b)
    # Enumerate 2,3-smooth a and 5,7-smooth b, combine coprime pairs.
    if time.time() < deadline - 15:
        # Generate 2,3-smooth numbers up to limit
        max_comp = 10**7  # each component up to 10^7, product up to 10^14
        smooth_23 = []
        a = 1
        while a <= max_comp:
            b = 1
            while a * b <= max_comp:
                smooth_23.append(a * b)
                b *= 3
            a *= 2

        # Generate 5,7-smooth numbers up to limit
        smooth_57 = []
        c = 1
        while c <= max_comp:
            d = 1
            while c * d <= max_comp:
                smooth_57.append(c * d)
                d *= 7
            c *= 5

        # Precompute sigma for smooth components
        def sigma_smooth_list(nums):
            return {n: sigma_exact(n) for n in nums}

        sigma_23 = sigma_smooth_list(smooth_23)
        sigma_57 = sigma_smooth_list(smooth_57)

        # For each smooth number, compute its "M contribution" = n * sigma(n)
        m_contrib_23 = {a: a * sigma_23[a] for a in smooth_23}
        m_contrib_57 = {b: b * sigma_57[b] for b in smooth_57}

        # Group m_contrib by value to find collisions among smooth components
        # But we need coprime pairs (a,b) with gcd(a,b)=1
        # Strategy: for each a in smooth_23, for each b in smooth_57 with gcd(a,b)==1:
        #   k = a*b, M = m_contrib_23[a] * m_contrib_57[b]
        # Group by M, count distinct k values.

        # To be efficient, group m_contrib_57 by value first
        m57_to_bs = defaultdict(list)
        for b in smooth_57:
            m57_to_bs[m_contrib_57[b]].append(b)

        m_to_ks = defaultdict(set)
        limit_k = 10**14  # k must be <= 10^14 to have M <= 10^18 (roughly)

        for a in smooth_23:
            if time.time() > deadline - 5:
                break
            ma = m_contrib_23[a]
            # For each distinct m57 value, check if any b is coprime to a
            for m57, bs in m57_to_bs.items():
                M = ma * m57
                if M > 10**18:
                    continue
                # Check if any b in bs is coprime to a and a*b is valid
                for b in bs:
                    if a * b > limit_k:
                        continue
                    if math.gcd(a, b) == 1:
                        k = a * b
                        if k <= math.isqrt(M):
                            m_to_ks[M].add(k)

        # Find best from MITM results
        for M, ks in m_to_ks.items():
            if len(ks) > best_count:
                best_count = len(ks)
                best_M = M
                best_k = list(ks)

    # ---------- Final verification ----------
    if best_M is not None and best_k:
        isqrt_M = math.isqrt(best_M)
        valid_k = []
        for k in best_k:
            if not isinstance(k, int) or isinstance(k, bool):
                continue
            if k < 1 or k > isqrt_M:
                continue
            try:
                if k * sigma_exact(k) == best_M:
                    valid_k.append(k)
            except Exception:
                continue
        # Deduplicate
        valid_k = list(set(valid_k))
        if len(valid_k) >= 2:
            best_k = valid_k
            best_count = len(valid_k)
        else:
            best_M = 333312
            best_k = [336, 372, 434]
            best_count = 3

    best_M = int(best_M)
    best_k = [int(k) for k in best_k]

    return {'M': best_M, 'k': best_k}
