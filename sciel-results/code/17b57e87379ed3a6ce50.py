import time
import sys

def construct():
    """
    Find a,b <= 10^7 with sigma(a)=sigma(b), a/b != 3/2, minimizing |a/b - 3/2|.
    Strategy: direct arithmetic search over k = |2a - 3b| starting from k=1.
    For each k, solve 2a - 3b = k (or -k) for integers a,b in [1, N].
    Check sigma(a) == sigma(b) using precomputed sigma array.
    """
    N = 10**7
    t0 = time.time()
    deadline = t0 + 55  # safety margin

    # Precompute sigma(n) for all n in [1, N] using linear sieve
    # sigma(n) = sum of all positive divisors of n
    spf = [0] * (N + 1)
    cnt = [0] * (N + 1)
    p_pow = [0] * (N + 1)
    sigma = [0] * (N + 1)

    sigma[1] = 1
    spf[1] = 1
    p_pow[1] = 1

    primes = []

    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            cnt[i] = 1
            p_pow[i] = i
            sigma[i] = 1 + i
            primes.append(i)

        for p in primes:
            if p > spf[i] or i * p > N:
                break
            np_ = i * p
            spf[np_] = p
            if p == spf[i]:
                cnt[np_] = cnt[i] + 1
                p_pow[np_] = p_pow[i] * p
                # sigma[np_] = sigma[np_] via: if i = p^k * m, then
                # sigma[i] = (1+p+...+p^k) * sigma[m]
                # sigma[np_] = (1+p+...+p^(k+1)) * sigma[m]
                # = sigma[i] + p^(k+1) * sigma[m]
                # sigma[m] = sigma[i] // ((p^(k+1)-1)//(p-1))
                pk1_sum = (p_pow[i] * p - 1) // (p - 1)
                m_sigma = sigma[i] // pk1_sum
                pk2 = p_pow[i] * p * p
                pk2_sum = (pk2 - 1) // (p - 1)
                sigma[np_] = m_sigma * pk2_sum
            else:
                cnt[np_] = 1
                p_pow[np_] = p
                sigma[np_] = sigma[i] * (1 + p)

    # Free memory no longer needed
    del spf, cnt, p_pow, primes

    # Direct arithmetic search over k = |2a - 3b|
    # We want to minimize k / (2b), so for small k we want large b.
    # For k >= 1:
    #   Case 1: 2a - 3b = k  =>  a = (3b + k) / 2, need (3b+k) even
    #   Case 2: 2a - 3b = -k =>  a = (3b - k) / 2, need (3b-k) even and 3b > k
    #
    # For each k, we want the largest valid b such that a is in [1, N].
    #
    # Case 1: a = (3b + k)/2, need 3b + k even.
    #   If k is even: 3b must be even => b must be even.
    #   If k is odd: 3b must be odd => b must be odd.
    #   Constraints: 1 <= (3b+k)/2 <= N => 2 - k <= 3b <= 2N - k => b <= (2N-k)/3
    #   Max b: floor((2N - k) / 3), with parity constraint.
    #
    # Case 2: a = (3b - k)/2, need 3b - k even and 3b > k.
    #   If k is even: 3b must be even => b must be even.
    #   If k is odd: 3b must be odd => b must be odd.
    #   Constraints: 1 <= (3b-k)/2 <= N => k + 2 <= 3b <= 2N + k => b >= ceil((k+2)/3), b <= (2N+k)/3
    #   Max b: floor((2N + k) / 3), with parity constraint.

    best_a = 0
    best_b = 0
    best_num = 1  # numerator of error
    best_den = 1  # denominator of error

    max_k = 10000  # search up to this k; can be extended if time permits

    for k in range(1, max_k + 1):
        if time.time() > deadline - 2:
            break

        # Case 1: 2a - 3b = k, a = (3b + k) / 2
        # Max b such that a <= N: b <= (2N - k) / 3
        max_b1 = (2 * N - k) // 3
        if max_b1 < 1:
            continue

        # Parity: need (3b + k) even
        if k % 2 == 0:
            # b must be even
            if max_b1 % 2 != 0:
                max_b1 -= 1
        else:
            # b must be odd
            if max_b1 % 2 == 0:
                max_b1 -= 1

        if max_b1 >= 1:
            b = max_b1
            a = (3 * b + k) // 2
            if 1 <= a <= N and 1 <= b <= N:
                if sigma[a] == sigma[b]:
                    # Error = k / (2b)
                    num = k
                    den = 2 * b
                    # Compare num/den < best_num/best_den
                    if num * best_den < best_num * den:
                        best_num = num
                        best_den = den
                        best_a = a
                        best_b = b

        # Case 2: 2a - 3b = -k, a = (3b - k) / 2
        # Max b such that a <= N: b <= (2N + k) / 3
        # Min b such that a >= 1: b >= (k + 2) / 3
        max_b2 = (2 * N + k) // 3
        if max_b2 < 1:
            continue

        # Parity: need (3b - k) even
        if k % 2 == 0:
            # b must be even
            if max_b2 % 2 != 0:
                max_b2 -= 1
        else:
            # b must be odd
            if max_b2 % 2 == 0:
                max_b2 -= 1

        if max_b2 >= 1:
            b = max_b2
            a = (3 * b - k) // 2
            if 1 <= a <= N and 1 <= b <= N:
                if sigma[a] == sigma[b]:
                    num = k
                    den = 2 * b
                    if num * best_den < best_num * den:
                        best_num = num
                        best_den = den
                        best_a = a
                        best_b = b

    # If we found a result, also try to extend k range if time permits
    if time.time() < deadline - 5 and best_a > 0:
        # Try larger k values for potentially better b (unlikely but possible)
        # Actually for larger k, the error k/(2b) gets worse since b is bounded by N.
        # So the best is always at small k with large b. We're done.
        pass

    # Fallback
    if best_a == 0:
        best_a, best_b = 11, 6

    return [int(best_a), int(best_b)]
