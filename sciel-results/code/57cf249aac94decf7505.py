"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
import numpy as np

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit using a^2 * b^3 form."""
    if limit < 1:
        return [1]
    powerful = set()
    powerful.add(1)
    
    # b^3 * a^2 <= limit
    # b up to limit^(1/3)
    max_b = int(limit ** (1/3)) + 1
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > limit:
            break
        # a^2 <= limit / b3
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    
    return sorted(powerful)

def _is_powerful(n):
    """Check if n is a powerful number using exact factorization."""
    if n == 1:
        return True
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            # Count exponent of d in temp
            exp = 0
            while temp % d == 0:
                temp //= d
                exp += 1
            if exp % 2 == 1:
                return False
        d += 1
    # If temp > 1, it's a prime factor with exponent 1
    if temp > 1:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Generate powerful numbers up to a limit
    # We need a+b = N where a,b are powerful, so a,b <= N/2 roughly
    # But we want N up to 10^18, so a,b can be up to 10^18
    # However, generating all powerful numbers up to 10^18 is infeasible
    # We need to be strategic
    
    # Strategy: Use FFT to count representations
    # Generate powerful numbers up to some limit L
    # Then use FFT to find N with many representations
    
    # Let's try different limits and see what works
    # The key insight: we want N where many pairs (a,b) of powerful numbers sum to N
    
    # For FFT approach:
    # Create array A where A[i] = 1 if i is powerful, else 0
    # Then A * A (convolution) gives count of pairs summing to each N
    # But we need a <= b, so we need to be careful
    
    # Let's try limit = 10^7 first
    limit = 10_000_000
    
    if time.time() > deadline:
        # Fallback to simple search
        powerful_small = _generate_powerful(10000)
        members = set(powerful_small)
        best = max(range(2, 10001), key=lambda n: sum(1 for a in powerful_small if 2 * a <= n and n - a in members))
        return {'N': best, 'pairs': [[a, best - a] for a in powerful_small if 2 * a <= best and best - a in members]}
    
    powerful = _generate_powerful(limit)
    powerful_arr = np.array(powerful, dtype=np.int64)
    M = len(powerful_arr)
    
    if time.time() > deadline:
        powerful_small = _generate_powerful(10000)
        members = set(powerful_small)
        best = max(range(2, 10001), key=lambda n: sum(1 for a in powerful_small if 2 * a <= n and n - a in members))
        return {'N': best, 'pairs': [[a, best - a] for a in powerful_small if 2 * a <= best and best - a in members]}
    
    # Create indicator array
    max_val = limit
    A = np.zeros(max_val + 1, dtype=np.float64)
    A[powerful_arr] = 1.0
    
    # FFT size: need at least 2*max_val for convolution
    fft_size = 1
    while fft_size < 2 * (max_val + 1):
        fft_size *= 2
    
    # Compute FFT
    A_fft = np.fft.fft(A, n=fft_size)
    conv = np.fft.ifft(A_fft * A_fft).real
    conv = np.round(conv).astype(np.int64)
    
    # conv[n] = number of ordered pairs (a,b) with a+b=n, a,b in powerful
    # We want unordered pairs with a <= b
    # For a != b: each unordered pair counted twice in conv
    # For a == b: counted once
    # So unordered count = (conv[n] + count_of_diagonal) / 2
    # where count_of_diagonal = 1 if n/2 is powerful and n is even, else 0
    
    # Find N with maximum unordered pairs
    # conv is indexed by sum, so conv[n] gives ordered pair count
    # We need n >= 2 and n <= 2*limit
    
    # Compute unordered counts
    max_sum = 2 * limit
    # conv has length fft_size, but we only care about indices 2 to max_sum
    conv_relevant = conv[2:max_sum+1]
    
    # For each n, unordered_count = (conv[n] + diag[n]) / 2
    # diag[n] = 1 if n is even and n//2 is powerful
    n_vals = np.arange(2, max_sum + 1)
    diag = np.zeros(max_sum - 1, dtype=np.int64)
    even_mask = (n_vals % 2 == 0)
    half_vals = n_vals[even_mask] // 2
    # Check if half_vals are in powerful
    powerful_set = set(powerful)
    diag[even_mask] = np.array([1 if h in powerful_set else 0 for h in half_vals], dtype=np.int64)
    
    unordered_counts = (conv_relevant + diag) // 2
    
    # Find best N
    best_idx = np.argmax(unordered_counts)
    best_N = int(n_vals[best_idx])
    best_count = int(unordered_counts[best_idx])
    
    if time.time() > deadline:
        # Return what we have
        pairs = []
        for a in powerful:
            b = best_N - a
            if a <= b and b in powerful_set and b <= limit:
                pairs.append([a, b])
        return {'N': best_N, 'pairs': pairs}
    
    # Verify and collect pairs
    pairs = []
    for a in powerful:
        b = best_N - a
        if a <= b and b in powerful_set:
            # Verify both are powerful
            if _is_powerful(a) and _is_powerful(b):
                pairs.append([a, b])
    
    # Sort pairs
    pairs.sort()
    
    # Verify the count
    # The FFT count might be slightly off due to floating point, so verify
    verified_pairs = []
    for a, b in pairs:
        if a + b == best_N and a <= b and _is_powerful(a) and _is_powerful(b):
            verified_pairs.append([a, b])
    
    return {'N': best_N, 'pairs': verified_pairs}
# EVOLVE-BLOCK-END
