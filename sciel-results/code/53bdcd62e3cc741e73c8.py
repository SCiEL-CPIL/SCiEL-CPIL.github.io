"""Structural change: multi-target FFT convolution with powerful-number generation."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
import numpy as np
from collections import defaultdict


def _generate_powerfuls(limit):
    """Generate all powerful numbers <= limit.
    A powerful number has the form a^2 * b^3 where b is square-free.
    """
    powerfuls = set()
    max_b = int(limit ** (1/3)) + 2
    sqrt_max_b = isqrt(max_b)
    primes = []
    for x in range(2, sqrt_max_b + 1):
        is_p = True
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                is_p = False
                break
        if is_p:
            primes.append(x)
    
    for b in range(1, max_b + 1):
        is_sq_free = True
        for p in primes:
            if p * p > b:
                break
            if b % (p * p) == 0:
                is_sq_free = False
                break
        if not is_sq_free:
            continue
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerfuls.add(val)
    return powerfuls


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Generate powerful numbers up to a moderate limit
    # L=10^7 gives ~10000-15000 powerful numbers, manageable for FFT
    L = 10**7
    
    powerfuls = _generate_powerfuls(L)
    powerfuls_sorted = sorted(powerfuls)
    powerfuls_set = set(powerfuls)
    n = len(powerfuls_sorted)
    
    if time.time() > deadline - 80:
        # Fallback
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Strategy: Use FFT convolution to count representations for multiple N values
    # Create indicator array f where f[x] = 1 if x is powerful
    # Then (f * f)[s] = number of ordered pairs (a,b) with a+b=s, both powerful
    # We want unordered pairs a<=b, so for s: count = (conv[s] + (1 if s even and s/2 powerful else 0)) / 2
    
    max_sum = 2 * L
    # FFT size: next power of 2 >= 2*max_sum
    fft_size = 1
    while fft_size < 2 * max_sum:
        fft_size *= 2
    
    # Create indicator array
    f = np.zeros(fft_size, dtype=np.float64)
    for p in powerfuls_sorted:
        f[p] = 1.0
    
    # FFT convolution
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=fft_size)
    
    # conv[s] ≈ number of ordered pairs (a,b) with a+b=s
    # For unordered pairs with a<=b: 
    #   if s is odd: count = conv[s] / 2
    #   if s is even: count = (conv[s] + (1 if s/2 is powerful else 0)) / 2
    
    # Find the top candidates by conv[s] value
    # We only care about s in [2, 2*L]
    conv_relevant = conv[2:2*L+1]
    
    # Get top 50 candidates
    top_indices = np.argsort(conv_relevant)[::-1][:50]
    top_Ns = top_indices + 2
    
    # Now verify and extract exact pairs for each candidate
    best_N = None
    best_pairs = []
    best_count = 0
    
    for N in top_Ns:
        N = int(N)
        if N < 2:
            continue
        
        # Extract exact pairs
        pairs = []
        for a in powerfuls_sorted:
            if a > N // 2:
                break
            b = N - a
            if b in powerfuls_set:
                pairs.append([a, b])
        
        count = len(pairs)
        if count > best_count:
            best_count = count
            best_N = N
            best_pairs = pairs
        
        # Time check
        if time.time() > deadline - 20:
            break
    
    # If we didn't find anything, try a simple fallback
    if best_N is None:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Also try to improve: check if we can find a better N by looking at nearby values
    # or by trying a different L
    
    # Try with a larger L if we have time
    if time.time() < deadline - 60:
        L2 = 3 * 10**7
        powerfuls2 = _generate_powerfuls(L2)
        powerfuls_sorted2 = sorted(powerfuls2)
        powerfuls_set2 = set(powerfuls2)
        n2 = len(powerfuls_sorted2)
        
        if n2 > n:
            max_sum2 = 2 * L2
            fft_size2 = 1
            while fft_size2 < 2 * max_sum2:
                fft_size2 *= 2
            
            f2 = np.zeros(fft_size2, dtype=np.float64)
            for p in powerfuls_sorted2:
                f2[p] = 1.0
            
            F2 = np.fft.rfft(f2)
            conv2 = np.fft.irfft(F2 * F2, n=fft_size2)
            conv2_relevant = conv2[2:2*L2+1]
            
            top_indices2 = np.argsort(conv2_relevant)[::-1][:50]
            top_Ns2 = top_indices2 + 2
            
            for N in top_Ns2:
                N = int(N)
                if N < 2:
                    continue
                
                pairs = []
                for a in powerfuls_sorted2:
                    if a > N // 2:
                        break
                    b = N - a
                    if b in powerfuls_set2:
                        pairs.append([a, b])
                
                count = len(pairs)
                if count > best_count:
                    best_count = count
                    best_N = N
                    best_pairs = pairs
                
                if time.time() > deadline - 20:
                    break
    
    if best_N is None or best_count == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    return {'N': int(best_N), 'pairs': best_pairs}

# EVOLVE-BLOCK-END
