import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill

    def time_remaining():
        return DEADLINE - (time.time() - start_time)

    # Segmented sieve for phi in [low, high]
    def compute_phi_segment(low, high):
        size = high - low + 1
        phi = list(range(low, high + 1))
        rem = list(range(low, high + 1))

        # Sieve primes up to sqrt(high)
        limit = int(math.isqrt(high)) + 1
        is_prime = [True] * (limit + 1)
        primes = []
        for i in range(2, limit + 1):
            if is_prime[i]:
                primes.append(i)
                for j in range(i * i, limit + 1, i):
                    is_prime[j] = False

        for p in primes:
            # Find first multiple of p in [low, high]
            first = ((low + p - 1) // p) * p
            for j in range(first, high + 1, p):
                idx = j - low
                if rem[idx] % p == 0:
                    phi[idx] = phi[idx] // p * (p - 1)
                    while rem[idx] % p == 0:
                        rem[idx] //= p

        # For any remaining composite factor > 1
        for i in range(size):
            if rem[i] > 1:
                p = rem[i]
                phi[i] = phi[i] // p * (p - 1)

        return phi

    # Find longest run of distinct phi values in a segment
    def find_longest_run(phi_vals, offset, max_k=200):
        """
        phi_vals: list of phi values for [offset, offset + len(phi_vals) - 1]
        offset: the actual number corresponding to phi_vals[0]
        Returns (best_k, best_start)
        """
        n = len(phi_vals)
        best_k = 0
        best_start = offset
        # Use a sliding window approach
        # We want to find the longest contiguous subsequence with all distinct values
        # Since max_k is 200, we can use a set and check window of size up to max_k
        from collections import defaultdict
        counts = defaultdict(int)
        left = 0
        for right in range(n):
            counts[phi_vals[right]] += 1
            # Shrink window until all distinct
            while counts[phi_vals[right]] > 1:
                counts[phi_vals[left]] -= 1
                left += 1
            window_len = right - left + 1
            if window_len > max_k:
                # We only care about up to max_k, but if window > max_k, 
                # the sub-window of size max_k is valid
                # Actually, if window_len > max_k, then any consecutive max_k 
                # elements within this window have distinct values
                # So we can update best with max_k
                if max_k > best_k:
                    best_k = max_k
                    best_start = offset + left + (window_len - max_k)
                    # Actually, let's just set best_k = max_k and best_start = offset + left
                    # since a window of size max_k starting at left is valid
                    best_start = offset + left
            elif window_len > best_k:
                best_k = window_len
                best_start = offset + left
        return best_k, best_start

    best_k = 1
    best_start = 1

    # Strategy: search in segments across the range [1, 10^12]
    # Start with small segments near 1, then expand
    # Use increasing segment sizes and strides

    # First, do a quick search near 1 with small segments
    segment_size = 100000
    low = 1
    while low <= 10**12 and time_remaining() > 5.0:
        high = min(low + segment_size - 1, 10**12)
        phi_vals = compute_phi_segment(low, high)
        k, s = find_longest_run(phi_vals, low, max_k=200)
        if k > best_k:
            best_k = k
            best_start = s
        low = high + 1
        # Increase segment size if we've found good results
        if best_k > 50:
            segment_size = min(segment_size * 2, 1000000)

    return {'start': best_start, 'length': best_k}
