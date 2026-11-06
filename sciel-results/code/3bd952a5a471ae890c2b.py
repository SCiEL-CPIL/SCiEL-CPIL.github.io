import time
import math

def construct():
    deadline = time.time() + 115

    def sieve_primes(limit):
        if limit < 2:
            return []
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_p[i]:
                is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
        return [i for i in range(2, limit + 1) if is_p[i]]

    # Precompute primes up to 5 million for trial division
    SIEVE_LIMIT = 5_000_000
    primes_list = sieve_primes(SIEVE_LIMIT)
    n_primes = len(primes_list)
    
    # Binary search helper to find index of first prime > x
    def prime_index_gt(x):
        lo, hi = 0, n_primes
        while lo < hi:
            mid = (lo + hi) // 2
            if primes_list[mid] <= x:
                lo = mid + 1
            else:
                hi = mid
        return lo
    
    def is_p_smooth(n, p):
        """Check if n is p-smooth (all prime factors <= p)."""
        if n <= 1:
            return True
        # Find how many primes to trial divide by (primes <= p)
        idx = prime_index_gt(p)
        # Also limit by sqrt(n) since after that, at most one factor remains
        limit = min(idx, n_primes)
        for i in range(limit):
            pr = primes_list[i]
            if pr * pr > n:
                break
            while n % pr == 0:
                n //= pr
            if n == 1:
                return True
        # After trial division, if n > 1, it must be prime (or 1)
        # If n > p, it has a prime factor > p, so not p-smooth
        return n <= p

    best_k = 0
    best_p = 2

    # Search over primes. We need p <= 10^9.
    # For small p, the range p^2+1 to p^2+k is small, so smoothness is easier.
    # Let's search primes in increasing order but with a time budget.
    
    # First pass: search primes up to SIEVE_LIMIT quickly
    for p in primes_list:
        if time.time() > deadline - 10:
            break
        base = p * p
        k = 0
        for i in range(1, 201):
            n = base + i
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
            if best_k >= 200:
                break

    # Second pass: if we have time, search larger primes
    # Generate primes between SIEVE_LIMIT and 10^9 using segmented sieve
    if time.time() < deadline - 20:
        SEG_SIZE = 1_000_000
        start = SIEVE_LIMIT + 1
        end = min(10**9, start + 10 * SEG_SIZE)  # Search up to 10^7 first
        
        # We need primes up to sqrt(end) for the segmented sieve
        sqrt_end = int(math.isqrt(end)) + 1
        small_primes = [pr for pr in primes_list if pr <= sqrt_end]
        
        seg_start = start
        while seg_start < end and time.time() < deadline - 15:
            seg_end = min(seg_start + SEG_SIZE - 1, end)
            seg = bytearray(b'\x01') * (seg_end - seg_start + 1)
            for pr in small_primes:
                first = ((seg_start + pr - 1) // pr) * pr
                if first == pr:
                    first += pr
                for j in range(first - seg_start, seg_end - seg_start + 1, pr):
                    seg[j] = 0
            for j in range(seg_end - seg_start + 1):
                if seg[j]:
                    p = seg_start + j
                    if time.time() > deadline - 10:
                        break
                    base = p * p
                    k = 0
                    for i in range(1, 201):
                        n = base + i
                        if is_p_smooth(n, p):
                            k += 1
                        else:
                            break
                    if k > best_k:
                        best_k = k
                        best_p = p
                        if best_k >= 200:
                            break
                    if best_k >= 200:
                        break
            seg_start = seg_end + 1
            if best_k >= 200:
                break

    return {'p': best_p, 'k': best_k}
