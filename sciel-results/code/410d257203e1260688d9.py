import time
import math
import numpy as np
from collections import defaultdict


def _factorize(n):
    """Factorize n by trial division."""
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


def _sigma_exact(n):
    """Compute sigma(n) exactly via factorization."""
    if n == 1:
        return 1
    factors = _factorize(n)
    result = 1
    for p, exp in factors:
        result *= (p ** (exp + 1) - 1) // (p - 1)
    return result


def _divisors_from_factors(factors):
    """Generate all divisors from prime factorization."""
    divs = [1]
    for p, exp in factors:
        new_divs = []
        for d in divs:
            val = d
            for _ in range(exp + 1):
                new_divs.append(val)
                val *= p
        divs = new_divs
    return divs


def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    # Fallback known solution
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3

    # Phase 1: Sieve for sigma up to N to find candidate M values
    N = 8_000_000
    if time.time() < deadline - 40.0:
        # Compute sigma using divisor-sum sieve
        sigma = np.zeros(N + 1, dtype=np.int64)
        for i in range(1, N + 1):
            sigma[i::i] += i

        # Count collisions: for each k, M = k * sigma[k]
        m_counts = {}
        for k in range(2, N + 1):
            M = int(k) * int(sigma[k])
            if M > 10**18:
                continue
            if M in m_counts:
                m_counts[M] += 1
            else:
                m_counts[M] = 1
            if time.time() > deadline - 35.0:
                break

        # Find top candidate M values
        if m_counts:
            # Sort by count descending
            sorted_ms = sorted(m_counts.items(), key=lambda x: -x[1])
            # Check top candidates
            for M_cand, cnt in sorted_ms[:5]:
                if cnt < best_count:
                    break
                if time.time() > deadline - 20.0:
                    break

                # For this M, enumerate all divisors and check k*sigma(k)=M
                try:
                    factors_M = _factorize(M_cand)
                    divs = _divisors_from_factors(factors_M)
                    sqrt_M = math.isqrt(M_cand)
                    valid_ks = []
                    for k in divs:
                        if k > sqrt_M:
                            break
                        if k < 2:
                            continue
                        # Quick check: k must divide M (guaranteed) and k*sigma(k)=M
                        # sigma(k) = M // k must be integer
                        sig_needed = M_cand // k
                        # Check if sigma(k) == sig_needed
                        # For k <= N, we have sigma[k] from sieve
                        if k <= N:
                            if int(sigma[k]) == sig_needed:
                                valid_ks.append(k)
                        else:
                            # Need exact computation
                            if time.time() > deadline - 10.0:
                                break
                            sig_val = _sigma_exact(k)
                            if sig_val == sig_needed:
                                valid_ks.append(k)

                    if len(valid_ks) > best_count:
                        best_count = len(valid_ks)
                        best_M = M_cand
                        best_k = sorted(valid_ks)
                except Exception:
                    continue

    # Phase 2: Also try known small M values and their divisor enumeration
    # The seed M=333312: let's verify and expand
    if time.time() < deadline - 10.0:
        for M_test in [333312, 666624, 999936]:
            if time.time() > deadline - 5.0:
                break
            try:
                factors_M = _factorize(M_test)
                divs = _divisors_from_factors(factors_M)
                sqrt_M = math.isqrt(M_test)
                valid_ks = []
                for k in divs:
                    if k > sqrt_M:
                        break
                    if k < 2:
                        continue
                    sig_needed = M_test // k
                    if k <= N and time.time() < deadline - 5.0:
                        if int(sigma[k]) == sig_needed:
                            valid_ks.append(k)
                    else:
                        if time.time() > deadline - 5.0:
                            break
                        sig_val = _sigma_exact(k)
                        if sig_val == sig_needed:
                            valid_ks.append(k)
                if len(valid_ks) > best_count:
                    best_count = len(valid_ks)
                    best_M = M_test
                    best_k = sorted(valid_ks)
            except Exception:
                continue

    # Phase 3: Targeted search - try to find more solutions for best_M
    # by checking divisors we might have missed
    if best_M is not None and time.time() < deadline - 5.0:
        try:
            factors_M = _factorize(best_M)
            divs = _divisors_from_factors(factors_M)
            sqrt_M = math.isqrt(best_M)
            valid_ks = []
            for k in divs:
                if k > sqrt_M:
                    break
                if k < 2:
                    continue
                sig_needed = best_M // k
                if time.time() > deadline - 2.0:
                    break
                sig_val = _sigma_exact(k)
                if sig_val == sig_needed:
                    valid_ks.append(k)
            if len(valid_ks) > best_count:
                best_count = len(valid_ks)
                best_k = sorted(valid_ks)
        except Exception:
            pass

    # Final verification
    if best_k:
        valid_k = []
        for k in best_k:
            if not isinstance(k, int) or isinstance(k, bool):
                continue
            if k < 1 or k > math.isqrt(best_M):
                continue
            try:
                if k * _sigma_exact(k) == best_M:
                    valid_k.append(k)
            except Exception:
                continue
        if len(valid_k) >= 2:
            best_k = sorted(set(valid_k))
            best_count = len(best_k)
        else:
            best_M = 333312
            best_k = [336, 372, 434]
            best_count = 3

    # Ensure types
    best_M = int(best_M)
    best_k = [int(k) for k in best_k]

    return {'M': best_M, 'k': best_k}
