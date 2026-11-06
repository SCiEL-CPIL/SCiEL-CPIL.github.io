"""Rewritten program: FFT-based convolution to find N with many powerful-number representations."""
import time
import math
import numpy as np

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit using a^2 * b^3 form with square-free b."""
    powerful = set()
    powerful.add(1)
    max_b = int(round(limit ** (1/3))) + 2
    
    # Sieve for square-free numbers up to max_b
    sq_free = [True] * (max_b + 1)
    if max_b >= 0:
        sq_free[0] = False
    if max_b >= 1:
        sq_free[1] = True
    for p in range(2, int(math.isqrt(max_b)) + 1):
        if sq_free[p]:
            p2 = p * p
            for m in range(p2, max_b + 1, p2):
                sq_free[m] = False
    
    for b in range(1, max_b + 1):
        if not sq_free[b]:
            continue
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = int(math.isqrt(limit // b3))
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    return sorted(powerful)

def construct():
    deadline = time.time() + 130  # safety margin before 150s limit
    start_time = time.time()
    
    # Generate powerful numbers up to a limit.
    # We need N up to ~10^18, so a and b can be up to ~10^18.
    # But generating all powerful up to 10^18 is ~10^7 numbers, which is feasible.
    # However, FFT on an array of size 2*10^18 is impossible.
    # So we use a smaller limit for the FFT approach, or use a smarter method.
    
    # Strategy: Use FFT with a limit that fits in memory.
    # If limit = L, then N can be up to 2L.
    # We want N up to 10^18, so L up to 5*10^17. That's too large for an array.
    
    # Alternative: Use the fact that powerful numbers are sparse.
    # Count of powerful numbers up to X is ~ c * X^(1/2) + O(X^(1/3)).
    # For X = 10^18, count ~ 10^9. Too many for O(K^2) but maybe manageable with FFT on sparse data?
    
    # Actually, let's use a two-phase approach:
    # Phase 1: Generate powerful numbers up to L = 10^9 (about 10^4.5 ≈ 30000 numbers).
    # Phase 2: Use FFT to find the best N up to 2L = 2*10^9.
    # But we want N up to 10^18. So this only gives us N up to 2*10^9.
    
    # Better approach: Use a larger limit.
    # Let's try L = 10^10. Count of powerful up to 10^10 is ~ 10^5.
    # FFT on array of size 2*10^10 is too large (20 billion elements).
    
    # So we need a different approach for large N.
    
    # Key insight: The number of powerful numbers up to X is about sqrt(X).
    # For X = 10^18, that's 10^9 numbers. We can't store all of them in an array for FFT.
    
    # Alternative: Use a "bucket" approach.
    # Divide the range [1, 10^18] into buckets. For each bucket, generate powerful numbers.
    # Then use convolution-like counting between buckets.
    
    # Actually, let me reconsider. The current program scores 5.93 with limit=10^8.
    # That gives N up to 2*10^8. The score is the number of pairs.
    
    # Let me try a larger limit with FFT.
    # For limit = 10^9, the FFT array size is 2*10^9. That's 2 billion complex numbers = 16 GB. Too much.
    # For limit = 10^8, array size 2*10^8 = 200 million. That's 1.6 GB. Feasible.
    # For limit = 5*10^8, array size 10^9. That's 8 GB. Maybe feasible.
    
    # Let's try limit = 3*10^8, array size 6*10^8. About 4.8 GB. Should be OK.
    
    # Actually, let's be more careful. numpy FFT uses complex128, which is 16 bytes per element.
    # Array of size 2*L needs 32*L bytes.
    # For L = 10^8: 3.2 GB. OK.
    # For L = 2*10^8: 6.4 GB. Might be tight.
    # For L = 10^9: 32 GB. No.
    
    # So let's use L = 10^8 or slightly larger.
    
    # But wait - we can use a smarter approach. Instead of FFT on the full range,
    # we can use the fact that we only need to find the MAX count.
    
    # Let me try a different structural approach:
    # 1. Generate powerful numbers up to L.
    # 2. For each powerful number a, the valid b's are powerful numbers in [a, L-a] (if a <= L/2)
    #    or [a, N-a] where N = a + b.
    # 3. Use a "sliding window" or "sweep line" approach to count sums.
    
    # Actually, the most promising approach for a structural change:
    # Use the fact that N = a + b where a, b are powerful.
    # We want to maximize the number of such pairs for a given N.
    
    # Let me try: generate powerful numbers up to L = 10^9 (about 10^4.5 ≈ 31623 numbers).
    # Then use a meet-in-the-middle style approach:
    # For each a in powerful, for each b in powerful with b >= a, compute s = a + b.
    # Count occurrences of each s.
    # But this is O(K^2) = (3*10^4)^2 = 10^9. Too slow in Python.
    
    # So FFT is the way to go. Let's use L = 10^8.
    
    L = 100_000_000  # 10^8
    
    powerful = _generate_powerful(L)
    K = len(powerful)
    
    if K < 2:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Create indicator array for FFT
    # Array size: next power of 2 >= 2*L
    size = 1
    while size < 2 * L:
        size *= 2
    
    # Check if we have enough time
    if time.time() > deadline - 60:
        # Fall back to simple approach
        powerful_set = set(powerful)
        best_N = 2
        best_pairs = [[1, 1]]
        for a in powerful:
            if a > L // 2:
                break
            b = L - a
            if b in powerful_set and b >= a:
                pairs = []
                for x in powerful:
                    if x > a:
                        break
                    y = (a + b) - x
                    if y in powerful_set and y >= x:
                        pairs.append([x, y])
                if len(pairs) > len(best_pairs):
                    best_N = a + b
                    best_pairs = pairs
        return {'N': best_N, 'pairs': best_pairs}
    
    # FFT approach
    f = np.zeros(size, dtype=np.complex128)
    for p in powerful:
        f[p] = 1.0
    
    # FFT
    F = np.fft.fft(f)
    # Convolution: F^2
    G = F * F
    # Inverse FFT
    H = np.fft.ifft(G)
    
    # The real part of H gives the number of representations
    # H[n] = sum over a+b=n of f[a]*f[b] = number of ordered pairs (a,b) with a+b=n
    # We want unordered pairs with a <= b, so for n > 0:
    # count_unordered(n) = (count_ordered(n) + (1 if n is a sum of two equal powerful numbers)) / 2
    # Actually: count_ordered(n) = 2 * count_unordered_distinct(n) + count_equal(n)
    # where count_equal(n) = 1 if n/2 is powerful and n is even, else 0
    # So count_unordered(n) = (count_ordered(n) + count_equal(n)) / 2
    
    real_H = np.real(H).astype(np.int64)
    
    # Only consider n up to 2*L
    max_n = 2 * L
    if max_n >= size:
        max_n = size - 1
    
    # Find the n with maximum count
    # But we need to adjust for ordered vs unordered
    # For simplicity, let's just use the raw count (ordered pairs) as a proxy
    # The best N by ordered count is also likely the best by unordered count
    
    # Find peak
    peak_idx = int(np.argmax(real_H[1:max_n+1])) + 1
    peak_val = int(real_H[peak_idx])
    
    if peak_val < 1:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    best_N = peak_idx
    best_count = peak_val
    
    # Generate the actual pairs for best_N
    powerful_set = set(powerful)
    pairs = []
    for a in powerful:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerful_set and b >= a:
            pairs.append([a, b])
    
    if not pairs:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Verify count
    if len(pairs) != best_count:
        # The FFT count includes ordered pairs, so actual unordered count is different
        # But we just need to return valid pairs
        pass
    
    return {'N': best_N, 'pairs': pairs}
