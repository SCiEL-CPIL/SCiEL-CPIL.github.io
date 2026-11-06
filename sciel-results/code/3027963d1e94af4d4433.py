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
    
    max_b = int(limit ** (1/3)) + 1
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    
    return sorted(powerful)

def _fft_convolve_counts(powerful_arr, max_sum):
    """Use FFT to count how many pairs (a,b) with a<=b from powerful_arr sum to each value up to max_sum."""
    M = len(powerful_arr)
    # Create indicator array
    size = max_sum + 1
    # Find next power of 2 for FFT efficiency
    n_fft = 1
    while n_fft < 2 * size:
        n_fft *= 2
    
    # Create signal
    signal = np.zeros(n_fft, dtype=np.float64)
    signal[powerful_arr] = 1.0
    
    # FFT convolution
    fft_signal = np.fft.rfft(signal)
    conv = np.fft.irfft(fft_signal * fft_signal, n=n_fft)
    
    # conv[k] = number of ordered pairs (a,b) such that a+b=k
    # We want unordered pairs with a<=b, so for k, count = conv[k] but we need to adjust
    # Actually for a+b=k with a<=b: if a != b, each unordered pair is counted twice in conv[k]
    # if a == b, it's counted once
    # So unordered_count[k] = (conv[k] + is_powerful[k//2] if k even) / 2
    # But simpler: just use conv[k] // 2 + (1 if k%2==0 and k//2 in powerful else 0)
    # For now, let's just return the ordered count and divide appropriately
    
    return conv[:max_sum+1].astype(np.int64)

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill
    
    best_result = {'N': 2, 'pairs': [[1, 1]]}
    best_count = 1
    
    # Try increasing limits for powerful number generation
    # The key structural change: use FFT to count representations for ALL sums at once
    # This allows us to search much larger ranges of N
    limits_to_try = [10**6, 5*10**6, 10**7, 5*10**7, 10**8, 5*10**8, 10**9]
    
    for limit in limits_to_try:
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
        
        powerful = _generate_powerful(limit)
        powerful_set = set(powerful)
        powerful_arr = np.array(powerful, dtype=np.int64)
        M = len(powerful_arr)
        
        if M < 2:
            continue
        
        # The maximum sum we can form is 2*limit, but N must be <= 10^18
        # We'll search for the best N in range [2, 2*limit]
        max_sum = 2 * limit
        
        # Check if FFT is feasible
        n_fft = 1
        while n_fft < 2 * (max_sum + 1):
            n_fft *= 2
        
        # Estimate memory: n_fft * 8 bytes for float64
        if n_fft > 2 * 10**8:  # ~1.6 GB, too much
            continue
        
        try:
            # Create indicator array
            signal = np.zeros(n_fft, dtype=np.float64)
            signal[powerful_arr] = 1.0
            
            # FFT convolution
            fft_signal = np.fft.rfft(signal)
            conv = np.fft.irfft(fft_signal * fft_signal, n=n_fft)
            
            # conv[k] = number of ordered pairs (a,b) with a+b=k
            # For unordered pairs with a<=b:
            # count[k] = (conv[k] + delta) / 2 where delta=1 if k is even and k/2 is powerful
            # But for finding the MAX, we can just look at conv[k] since the adjustment is small
            # Actually let's compute properly
            # For each k, unordered_count = (conv[k] + (1 if k%2==0 and k//2 in powerful_set else 0)) // 2
            # But this is expensive to compute for all k. Let's just use conv[k] as a proxy
            # and verify the best candidate
            
            # Find the k with maximum conv[k] in range [2, max_sum]
            # conv[0] and conv[1] are not valid (N>=2)
            if max_sum + 1 > n_fft:
                conv_valid = conv[:n_fft]
            else:
                conv_valid = conv[:max_sum+1]
            
            # Zero out indices 0 and 1
            if len(conv_valid) > 2:
                conv_valid[:2] = 0
            
            if len(conv_valid) == 0:
                continue
            
            best_idx = int(np.argmax(conv_valid))
            best_N = int(best_idx)
            
            if best_N < 2:
                continue
            
            # Now compute the actual unordered pair count for best_N
            # Count pairs (a, b) with a <= b, a + b = best_N, both powerful
            actual_count = 0
            for a in powerful:
                if 2 * a > best_N:
                    break
                b = best_N - a
                if b in powerful_set and a <= b:
                    actual_count += 1
            
            if actual_count > best_count:
                best_count = actual_count
                pairs = []
                for a in powerful:
                    if 2 * a > best_N:
                        break
                    b = best_N - a
                    if b in powerful_set and a <= b:
                        pairs.append([a, b])
                best_result = {'N': best_N, 'pairs': pairs}
                
        except (MemoryError, Exception):
            continue
        
        elapsed = time.time() - start_time
        if elapsed > deadline - 20:
            break
    
    return best_result

# EVOLVE-BLOCK-END
