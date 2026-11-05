import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # Generate a few candidate sequences
    candidates = []
    
    # 1. Seed
    candidates.append(np.ones(N))
    
    # 2. Zadoff-Chu like sequences (Cosine chirp)
    # h_i = cos(2*pi*q*i*(i+1)/N + phi)
    # Try a few q and phi
    
    for q in [1, 3, 7, 13, 23, 47, 101, 211]:
        if q % N == 0: continue
        for phi in [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/4]:
            i = np.arange(N)
            phase = 2 * np.pi * q * i * (i + 1) / N + phi
            h_zc = np.cos(phase)
            candidates.append(h_zc)
            
    # 3. Add DC offsets to the ZC sequences to increase Sum
    # We can optimize alpha for each base sequence
    # Value(alpha) = 2*N*max(conv(base+alpha*1, base+alpha*1)) / (sum(base)+alpha*N)^2
    
    best_h = candidates[0]
    best_val = np.inf
    
    for base in candidates:
        if time.time() > deadline:
            break
            
        # Optimize alpha
        # Let S_base = sum(base), C_base = conv(base, base)
        # h = base + alpha * ones
        # Sum(h) = S_base + alpha * N
        # conv(h,h) = conv(base,base) + alpha * (conv(base, ones) + conv(ones, base)) + alpha^2 * conv(ones, ones)
        # conv(base, ones) is a "cumulative sum" like thing.
        # conv(ones, ones) is a triangle with peak N.
        
        # We can just sample alpha
        # alpha range: -2 to 2
        alphas = np.linspace(-2.0, 2.0, 100)
        
        S_base = np.sum(base)
        conv_base = np.convolve(base, base)
        
        # Precompute terms for speed?
        # conv(h,h) = conv_base + alpha * (conv(base, ones) + conv(ones, base)) + alpha^2 * conv(ones, ones)
        # conv(base, ones) = np.convolve(base, np.ones(N))
        conv_base_ones = np.convolve(base, np.ones(N))
        conv_ones_base = conv_base_ones # symmetric
        
        conv_ones_ones = np.convolve(np.ones(N), np.ones(N))
        
        for alpha in alphas:
            S_h = S_base + alpha * N
            if abs(S_h) < 1e-6:
                continue
            
            conv_h = conv_base + alpha * (conv_base_ones + conv_ones_base) + (alpha ** 2) * conv_ones_ones
            max_conv = np.max(conv_h)
            
            val = abs(2 * N * max_conv / (S_h ** 2))
            
            if val < best_val:
                best_val = val
                best_h = base + alpha * np.ones(N)
                
    # Final check
    if time.time() > deadline:
         return np.ones(N)
         
    return best_h
