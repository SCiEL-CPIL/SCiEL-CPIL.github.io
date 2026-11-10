"""Program using segmented sieve to compute phi for a large block and find max run of distinct totients."""
import numpy as np

def construct():
    # We need to find a contiguous run of integers n, n+1, ..., n+k-1
    # such that phi(n), phi(n+1), ..., phi(n+k-1) are pairwise distinct.
    # Maximize k, with n >= 1, k <= 200, n + k - 1 <= 10^12.
    
    # Strategy: Compute phi for a large contiguous block [L, R] using a segmented sieve.
    # Then scan for the longest run of distinct phi values.
    
    # The block size should be large enough to likely contain the maximum run,
    # but small enough to fit in memory and compute within 120s.
    # Let's use a block of size ~2 * 10^7.
    
    block_size = 20_000_000  # 2 * 10^7
    
    # We need to pick a starting point L. Since the maximum run is likely not at the very beginning,
    # and we want L + block_size - 1 <= 10^12, we can pick L from various positions.
    # However, computing phi for 2*10^7 numbers requires primes up to sqrt(R).
    # If R is up to 10^12, sqrt(R) is up to 10^6.
    # Sieve of Eratosthenes up to 10^6 is fast.
    
    # Let's pick a specific L. To maximize the chance of finding a good run,
    # we might want to search in a region where numbers are "typical".
    # Actually, the distribution of phi values is somewhat uniform in log scale,
    # but collisions become more likely as values get larger.
    # However, for a fixed k, the probability that k random numbers have distinct phi values
    # depends on the range of phi values.
    # phi(n) is roughly n * e^{-gamma} / log log n for large n.
    # For n ~ 10^12, phi(n) ~ 10^12 * 0.56 / log(log(10^12)) ~ 10^12 * 0.56 / 2.5 ~ 2*10^11.
    # The number of possible phi values in a window of size k is small compared to the range,
    # so collisions are rare. The limiting factor is likely the structure of phi(n) for consecutive n.
    
    # Let's try a few different starting positions to be safe, or just one large block.
    # Given the 120s limit, let's do one large block computation.
    
    # Let's choose L such that the block is in the middle of the range, e.g., L = 10^11.
    # But we need primes up to sqrt(L + block_size).
    # Let's set R = 10^12 - 1, so L = R - block_size + 1.
    # This ensures we're using the largest possible numbers, which might have "better" properties?
    # Or maybe smaller numbers are easier?
    # Actually, the problem says "Maximize k". The maximum k is a fixed number.
    # We just need to find it.
    
    # Let's try L = 10^12 - block_size.
    R = 10**12
    L = R - block_size + 1
    
    # Sieve of Eratosthenes up to sqrt(R) = 10^6
    limit = int(np.sqrt(R)) + 1
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    # Initialize phi array for the block [L, R]
    # phi[i] will store phi(L + i)
    # Initially, phi[i] = L + i
    phi_arr = np.arange(L, R + 1, dtype=np.int64)
    
    # For each prime p, find multiples of p in [L, R] and apply the phi reduction
    for p in primes:
        # Find the first multiple of p in [L, R]
        first_mult = ((L + p - 1) // p) * p
        if first_mult > R:
            continue
        
        # For each multiple m = k*p, phi(m) = phi(m) / p * (p-1)
        # We iterate over multiples
        for m in range(first_mult, R + 1, p):
            idx = m - L
            if phi_arr[idx] % p == 0:
                phi_arr[idx] = phi_arr[idx] // p * (p - 1)
    
    # Now phi_arr contains phi(L), phi(L+1), ..., phi(R)
    # We need to find the longest contiguous subsequence with distinct values.
    # We can use a sliding window with a hash set.
    
    # Since k is at most 200, we can just check all windows of size up to 200.
    # But a sliding window is more efficient.
    
    best_k = 1
    best_s = L
    
    # Use a set to track values in the current window
    # Window [left, right]
    left = 0
    seen = set()
    
    for right in range(len(phi_arr)):
        val = phi_arr[right]
        # If val is already in seen, shrink from left
        while val in seen:
            seen.remove(phi_arr[left])
            left += 1
        seen.add(val)
        current_len = right - left + 1
        if current_len > best_k:
            best_k = current_len
            best_s = L + left
        # Cap the window size at 200? No, we want to find the max, but k <= 200.
        # If current_len > 200, we can stop expanding this window?
        # Actually, if we find a run longer than 200, the max is at least 200.
        # But the problem says k <= 200. So if we find a run of 201, we can just return 200.
        # However, we should check if there's a run of exactly 200 or more.
        # Since we want to maximize k, and k is capped at 200, if we ever find a window of size >= 200,
        # we can immediately return 200 (provided the start is valid).
        if current_len >= 200:
            # We found a run of at least 200.
            # The start of this window is L + left.
            # We need to ensure L + left + 200 - 1 <= 10^12.
            # Since R = 10^12, and right <= R-L, left <= right, L + left + 199 <= L + right <= R = 10^12.
            return {'start': L + left, 'length': 200}
    
    # If we didn't find a run of 200, return the best we found.
    # Note: best_k might be less than 200.
    return {'start': best_s, 'length': best_k}
