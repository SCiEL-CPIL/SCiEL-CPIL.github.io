import time
import math
import numpy as np
from collections import defaultdict

def is_square_free(n):
    """Check if n is square-free."""
    if n < 2:
        return True
    for p in range(2, int(math.isqrt(n)) + 1):
        if n % (p * p) == 0:
            return False
    return True

def generate_powerful(limit):
    """Generate all powerful numbers up to limit."""
    powerful = set()
    # n = x^2 * y^3
    # y must be square-free
    max_y = int(round(limit ** (1.0 / 3.0))) + 2
    for y in range(1, max_y + 1):
        if not is_square_free(y):
            continue
        y3 = y ** 3
        if y3 > limit:
            break
        max_x = int(math.isqrt(limit // y3))
        for x in range(1, max_x + 1):
            val = x * x * y3
            if val <= limit:
                powerful.add(val)
            else:
                break
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Strategy 1: FFT-based search for N up to ~2*10^7
    # We can try to increase the limit if time permits, but 10^7 is safe.
    L = 10**7
    try:
        # 1. Generate powerful numbers
        t0 = time.time()
        powerful_list = generate_powerful(L)
        powerful_set = set(powerful_list)
        t1 = time.time()
        
        if time.time() > deadline - 10:
            raise TimeoutError("Generation too slow")

        # 2. Setup FFT
        # Max sum is 2*L. We need array size >= 2*L.
        # Next power of 2
        n_fft = 1
        while n_fft < 2 * L:
            n_fft *= 2
            
        # Check memory feasibility roughly
        # n_fft * 16 bytes (complex) 
        if n_fft > 2**26: # 67M complex ~ 1GB
            # Fallback to smaller L or brute force
            L = 10**6
            powerful_list = generate_powerful(L)
            powerful_set = set(powerful_list)
            n_fft = 1
            while n_fft < 2 * L:
                n_fft *= 2

        # Create signal
        # Use float64 for FFT
        sig = np.zeros(n_fft, dtype=np.float64)
        for p in powerful_list:
            sig[p] = 1.0
            
        # FFT
        t2 = time.time()
        if time.time() > deadline - 20:
            raise TimeoutError("FFT setup too slow")
            
        sig_fft = np.fft.fft(sig)
        # Square the spectrum
        sig_fft_sq = sig_fft * sig_fft
        # Inverse FFT
        conv = np.fft.ifft(sig_fft_sq).real
        
        # Round to nearest integer
        # Convolution coefficients are counts of pairs (a,b) such that a+b=i
        # Note: FFT convolution is cyclic, but we padded enough so it's linear
        counts = np.round(conv).astype(np.int64)
        
        # Find max count
        # We only care about N >= 2
        max_count = 0
        best_N = 0
        # Search in the valid range
        # The counts are valid up to 2*L
        search_limit = min(2 * L, n_fft)
        
        # Optimize max finding
        # counts[2:search_limit]
        if search_limit > 2:
            segment = counts[2:search_limit]
            max_idx = np.argmax(segment)
            max_count = segment[max_idx]
            best_N = max_idx + 2
            
        t3 = time.time()
        
        # 3. Verify and extract pairs for best_N
        if max_count == 0:
            # Fallback if FFT failed or found nothing
            best_N = 10
            max_count = 0
            
        # Generate pairs for best_N
        pairs = []
        for a in powerful_list:
            if a > best_N // 2:
                break
            b = best_N - a
            if b in powerful_set:
                pairs.append([a, b])
                
        # Sort pairs
        pairs.sort(key=lambda x: x[0])
        
        # Final check
        if len(pairs) < max_count:
             # Discrepancy, trust the explicit check
             max_count = len(pairs)
             
        return {'N': best_N, 'pairs': pairs}
        
    except Exception as e:
        # Fallback to brute force for smaller range if FFT fails
        try:
            L_fb = 20000
            powerful_fb = generate_powerful(L_fb)
            powerful_set_fb = set(powerful_fb)
            best_N = 0
            best_count = 0
            best_pairs = []
            
            for i, a in enumerate(powerful_fb):
                for b in powerful_fb[i:]:
                    s = a + b
                    if s > L_fb * 2:
                        break
                    # This brute force is O(M^2) which is slow for counting all
                    # Better: for each N, count pairs.
                    # But we just need the best.
                    # Let's just reuse the seed logic for fallback
                    pass
            
            # Simple seed fallback
            limit = 20000
            powerful = sorted({a * a * b ** 3 for b in range(1, 28) for a in range(1, math.isqrt(limit // b ** 3) + 1)})
            members = set(powerful)
            best = max(range(2, limit + 1), key=lambda n: sum(1 for a in powerful if 2 * a <= n and n - a in members))
            return {'N': best, 'pairs': [[a, best - a] for a in powerful if 2 * a <= best and best - a in members]}
        except:
            return {'N': 10, 'pairs': [[1, 9]]}
