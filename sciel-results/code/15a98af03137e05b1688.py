import time
from math import isqrt
from collections import defaultdict

def _sigma_by_factorization(n):
    """Compute sigma(n) by trial factorization."""
    if n == 1:
        return 1
    result = 1
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            power_sum = 1
            power_val = 1
            while temp % p == 0:
                temp //= p
                power_val *= p
                power_sum += power_val
            result *= power_sum
        p += 1 if p == 2 else 2
    if temp > 1:
        result *= (1 + temp)
    return result

def construct():
    start = time.time()
    deadline = start + 110.0  # Safety margin before 150s kill

    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3

    # Phase 1: Sieve up to N=10M to find collisions among small k
    N = 10_000_000
    try:
        # Compute sigma using a linear-ish sieve approach
        # Use a list-based approach for memory efficiency
        sigma = [0] * (N + 1)
        for i in range(1, N + 1):
            for j in range(i, N + 1, i):
                sigma[j] += i

        # Group by M = k * sigma(k)
        m_to_k = defaultdict(list)
        for k in range(2, N + 1):
            M = k * sigma[k]
            if M <= 10**18:
                m_to_k[M].append(k)

        del sigma

        # Find the M with the most k values
        for M, ks in m_to_k.items():
            if len(ks) > best_count:
                best_count = len(ks)
                best_M = M
                best_k = ks[:]

        del m_to_k

    except MemoryError:
        pass
    except Exception:
        pass

    # Phase 2: Targeted search using smooth numbers for larger k
    if time.time() < deadline - 30:
        # Generate smooth numbers (products of small primes) up to ~2*10^7
        # These are more likely to have interesting sigma properties
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        max_smooth = 20_000_000

        # Use a priority queue / heap approach to generate smooth numbers
        import heapq
        smooth_set = set()
        heap = [1]
        smooth_set.add(1)

        while heap:
            val = heapq.heappop(heap)
            if val > max_smooth:
                continue
            for p in primes:
                new_val = val * p
                if new_val <= max_smooth and new_val not in smooth_set:
                    smooth_set.add(new_val)
                    heapq.heappush(heap, new_val)

        smooth_nums = sorted(smooth_set)
        del smooth_set
        del heap

        # For smooth numbers beyond our sieve range, compute sigma and check for collisions
        # We need to track M values that already have solutions from the sieve
        # Since we deleted m_to_k, we'll build a new one for smooth numbers only
        # and also re-check against known best_M

        m_to_k2 = defaultdict(list)
        for k in smooth_nums:
            if time.time() > deadline - 15:
                break
            if k <= N:
                continue  # Already covered by sieve
            if k > isqrt(10**18):  # k must be <= sqrt(M) and M <= 10^18
                continue
            try:
                sig = _sigma_by_factorization(k)
                M = k * sig
                if M <= 10**18 and k <= isqrt(M):
                    m_to_k2[M].append(k)
            except Exception:
                continue

        del smooth_nums

        # Check for new best from smooth numbers
        for M, ks in m_to_k2.items():
            if len(ks) > best_count:
                best_count = len(ks)
                best_M = M
                best_k = ks[:]

        # Also check if any M in m_to_k2 matches our current best_M
        if best_M in m_to_k2:
            existing = set(best_k)
            for k in m_to_k2[best_M]:
                if k not in existing:
                    best_k.append(k)
                    existing.add(k)
            best_count = len(best_k)

        del m_to_k2

    # Phase 3: Enumerate divisors of best_M to find additional solutions
    if time.time() < deadline - 5 and best_M > 0:
        try:
            # Factorize best_M
            temp_M = best_M
            factors_M = []
            p = 2
            while p * p <= temp_M:
                if temp_M % p == 0:
                    cnt = 0
                    while temp_M % p == 0:
                        temp_M //= p
                        cnt += 1
                    factors_M.append((p, cnt))
                p += 1 if p == 2 else 2
            if temp_M > 1:
                factors_M.append((temp_M, 1))

            # Generate all divisors
            divisors = [1]
            for p, exp in factors_M:
                new_divs = []
                for d in divisors:
                    for e in range(exp + 1):
                        new_divs.append(d * (p ** e))
                divisors = new_divs

            # For each divisor k of best_M, check if k * sigma(k) == best_M
            # and k <= isqrt(best_M)
            isqrt_M = isqrt(best_M)
            extra_ks = []
            for k in divisors:
                if 1 < k <= isqrt_M:
                    try:
                        if k * _sigma_by_factorization(k) == best_M:
                            extra_ks.append(k)
                    except Exception:
                        continue

            # Combine with existing best_k
            all_ks = list(set(best_k + extra_ks))
            if len(all_ks) > best_count:
                best_count = len(all_ks)
                best_k = all_ks

        except Exception:
            pass

    # Final verification
    if best_M is not None and best_k:
        valid_k = []
        for k in best_k:
            if not isinstance(k, int) or isinstance(k, bool):
                continue
            if k < 1 or k > isqrt(best_M):
                continue
            try:
                if k * _sigma_by_factorization(k) == best_M:
                    valid_k.append(k)
            except Exception:
                continue

        if len(valid_k) >= 2:
            best_k = valid_k
            best_count = len(valid_k)
        else:
            # Fallback
            best_M = 333312
            best_k = [336, 372, 434]
            best_count = 3

    # Ensure types are correct
    best_M = int(best_M)
    best_k = [int(k) for k in best_k]
    best_k = sorted(set(best_k))

    return {'M': best_M, 'k': best_k}
