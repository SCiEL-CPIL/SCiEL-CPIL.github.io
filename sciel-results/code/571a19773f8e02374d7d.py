"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
import numpy as np

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit using efficient enumeration."""
    powerful_set = set()
    powerful_set.add(1)
    
    sqrt_limit = isqrt(limit)
    
    for m in range(1, sqrt_limit + 1):
        temp = m
        primes = []
        for p in range(2, isqrt(temp) + 1):
            if temp % p == 0:
                primes.append(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            primes.append(temp)
        
        n_primes = len(primes)
        for mask in range(1 << n_primes):
            b = 1
            for i in range(n_primes):
                if mask & (1 << i):
                    b *= primes[i]
            if b > limit // (m * m):
                continue
            powerful_set.add(m * m * b)
    
    return sorted(powerful_set)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Use FFT-based convolution to count representations
    # f(N) = #{(a,b) : a+b=N, a,b powerful}
    # This is the coefficient of x^N in (sum x^a)^2
    # We can compute this via FFT in O(N log N)
    
    # First, determine a good N_MAX. The number of powerful numbers up to N
    # is roughly c*sqrt(N). For N=10^8, that's about 10000-20000.
    # FFT of size 2*N_MAX takes O(N_MAX log N_MAX) which for N_MAX=10^8 is ~10^8*27
    # That might be too slow. Let's try N_MAX = 5*10^7 first.
    
    N_MAX = 50_000_000
    
    # Check if we have time
    if time.time() - start_time > 5:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Generate powerful numbers
    powerful = _generate_powerful(N_MAX)
    powerful_set = set(powerful)
    P = len(powerful)
    
    if time.time() - start_time > deadline - 30:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Create indicator array
    # f[n] = 1 if n is powerful, else 0
    # Then g = f * f (convolution), g[N] = number of ordered pairs (a,b) with a+b=N
    # We want unordered pairs with a <= b, so for N, the count is (g[N] + (1 if N is powerful and N/2 is integer and N/2 is powerful else 0)) // 2
    # Actually: g[N] counts all ordered pairs. Unordered with a<=b: 
    #   if N is even and N/2 is powerful, then (g[N]+1)//2, else g[N]//2
    
    # Use FFT for convolution
    # Need size at least 2*N_MAX + 1
    size = 1
    while size < 2 * N_MAX + 1:
        size *= 2
    
    if time.time() - start_time > deadline - 20:
        # Fallback to brute force for small range
        N_SMALL = 5_000_000
        powerful_small = _generate_powerful(N_SMALL)
        powerful_small_set = set(powerful_small)
        import bisect
        from collections import defaultdict
        count = defaultdict(int)
        for i, a in enumerate(powerful_small):
            if 2 * a > N_SMALL:
                break
            lo = bisect.bisect_left(powerful_small, a)
            hi = bisect.bisect_right(powerful_small, N_SMALL - a)
            for j in range(lo, hi):
                count[a + powerful_small[j]] += 1
        if not count:
            return {'N': 2, 'pairs': [[1, 1]]}
        best_N = max(count, key=lambda n: count[n])
        pairs = []
        for a in powerful_small:
            if 2 * a > best_N:
                break
            b = best_N - a
            if b >= a and b in powerful_small_set:
                pairs.append([a, b])
        return {'N': best_N, 'pairs': pairs}
    
    # Build indicator array
    f = np.zeros(size, dtype=np.float64)
    for p in powerful:
        if p < size:
            f[p] = 1.0
    
    # FFT convolution
    F = np.fft.fft(f)
    G = F * F
    g = np.fft.ifft(G).real
    
    # g[N] should be close to an integer
    # Round to get counts
    g_int = np.round(g).astype(np.int64)
    
    # For each N, compute number of unordered pairs with a <= b
    # g[N] = number of ordered pairs (a,b) with a+b=N, a,b in powerful
    # Unordered with a<=b: 
    #   Let c = g[N] (ordered count)
    #   If N is even and N/2 is powerful, then one of the pairs is (N/2, N/2) which is self-paired
    #   So unordered count = (c + 1) // 2 if (N even and N/2 powerful) else c // 2
    #   Wait, that's not quite right. Let me think again.
    #   
    #   Ordered pairs: all (a,b) with a+b=N, a,b powerful
    #   Unordered with a<=b: 
    #     For each unordered pair {a,b} with a<b, it contributes 2 to ordered count
    #     For the pair {N/2, N/2} (if N even and N/2 powerful), it contributes 1 to ordered count
    #   So: ordered = 2*(#unordered with a<b) + (1 if self-pair exists)
    #       unordered_total = #unordered with a<b + (1 if self-pair exists)
    #   Therefore: unordered_total = (ordered + 1) // 2 if self-pair exists, else ordered // 2
    #   Which is: (ordered + (1 if self_pair else 0)) // 2... no.
    #   
    #   Let s = 1 if (N even and N//2 in powerful_set) else 0
    #   unordered = (ordered + s) // 2  -- this is wrong
    #   
    #   Actually: ordered = 2*(unordered - s) + s = 2*unordered - s
    #   So unordered = (ordered + s) // 2
    
    # We want to find N that maximizes unordered count
    # But we also need N <= N_MAX and N >= 2
    
    # Compute self-pair indicator for all N
    # N is even and N//2 is powerful
    half_powerful = np.zeros(N_MAX + 1, dtype=np.int8)
    for p in powerful:
        if 2 * p <= N_MAX:
            half_powerful[2 * p] = 1
    
    # unordered[N] = (g_int[N] + half_powerful[N]) // 2
    # But we only care about N from 2 to N_MAX
    # And g_int might have small numerical errors, so let's be careful
    
    # Find the N with maximum unordered count
    # To avoid issues, let's just look at the top candidates from g_int
    # and verify them
    
    # Get top 100 candidates by g_int value
    # But we need to consider the self-pair adjustment
    # Let's compute a score: score[N] = g_int[N] + half_powerful[N]
    # Then unordered = score[N] // 2
    
    score_arr = g_int[:N_MAX+1].copy()
    score_arr += half_powerful[:N_MAX+1]
    
    # Find top candidates
    # We want to find N that maximizes score_arr[N] // 2
    # Just find the max of score_arr
    
    # But numerical errors in FFT might cause issues. Let's find top 50 by score_arr
    # and verify each one by brute force
    
    top_indices = np.argsort(score_arr)[::-1][:50]
    
    best_N = 2
    best_count = 1
    best_pairs = [[1, 1]]
    
    for N_candidate in top_indices:
        N_candidate = int(N_candidate)
        if N_candidate < 2:
            continue
        if time.time() - start_time > deadline - 5:
            break
        
        # Verify by brute force: count pairs (a,b) with a<=b, a+b=N_candidate, a,b powerful
        pairs = []
        for a in powerful:
            if a > N_candidate // 2:
                break
            b = N_candidate - a
            if b < a:
                continue
            if b in powerful_set:
                pairs.append([a, b])
        
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_N = N_candidate
            best_pairs = pairs
    
    return {'N': best_N, 'pairs': best_pairs}
# EVOLVE-BLOCK-END
