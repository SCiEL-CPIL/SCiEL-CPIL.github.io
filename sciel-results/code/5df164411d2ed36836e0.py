import time
import numpy as np
from math import isqrt

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit."""
    powerful = set([1])
    # b^3 part
    b = 1
    while b**3 <= limit:
        b3 = b**3
        # a^2 part
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a*a * b3
            if val <= limit:
                powerful.add(val)
        b += 1
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    # Choose limit L. 
    # L=10^8 requires FFT size ~2^27. 
    # Let's try L=10^7 first to be safe on memory/time, then scale up if possible?
    # No, we need one run. Let's pick L=10^7 which is very safe.
    # Actually, let's try L=5 * 10^7.
    L = 10**7
    
    # Generate powerful numbers
    powerful = _generate_powerful(L)
    members = set(powerful)
    
    # Create indicator array
    # Size must be at least 2*L for linear convolution
    # Next power of 2
    size = 1
    while size < 2 * L:
        size *= 2
        
    indicator = np.zeros(size, dtype=np.float64)
    for p in powerful:
        indicator[p] = 1.0
        
    # FFT
    f = np.fft.fft(indicator)
    # Convolution is ifft(fft * fft)
    # Note: numpy fft returns complex. 
    # For real input, f[k] * f[k] is not the power spectrum (which is |f|^2).
    # We want linear convolution. 
    # Theorem: FFT(x * y) = FFT(x) * FFT(y).
    # So ifft(FFT(I) * FFT(I)) gives circular convolution.
    # With zero padding to size >= 2L, circular conv == linear conv.
    conv = np.fft.ifft(f * f)
    
    # conv[i] is the number of pairs (a,b) such that a+b = i (mod size).
    # Since we padded, for i < 2L, it is the exact count.
    # Take real part and round
    counts = conv.real[:2*L]
    counts = np.round(counts).astype(np.int64)
    
    # Find N with max count
    # N must be >= 2 and <= 10^18.
    # Our counts are for sums up to 2L.
    # We want to maximize the score (count).
    # We should pick the N with the highest count.
    # Note: The problem asks for N up to 10^18. Our N is <= 2L = 2*10^7, which is fine.
    
    max_count = 0
    best_N = 2
    for i in range(2, len(counts)):
        if counts[i] > max_count:
            max_count = counts[i]
            best_N = i
            
    # Verify and extract pairs
    pairs = []
    for a in powerful:
        if a > best_N:
            break
        b = best_N - a
        if b >= a and b in members:
            pairs.append([a, b])
            
    # Check time
    if time.time() > deadline:
        # Fallback to a simple valid result if we somehow took too long (unlikely for this size)
        return {'N': 2, 'pairs': [[1, 1]]}
        
    return {'N': best_N, 'pairs': pairs}
