import numpy as np
import time

N = 600
H = 0.5 / N
X = -0.25 + (np.arange(N) + 0.5) * H

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    def compute_score(heights):
        n_fft = 2048
        F = np.fft.rfft(heights, n=n_fft)
        conv = np.fft.irfft(F * F, n=n_fft) * H
        max_conv = np.max(conv)
        total_mass = np.sum(heights) * H
        return max_conv / (total_mass ** 2)

    def normalize(f):
        s = np.sum(f)
        if s == 0:
            return f
        return f / s

    def convolve(f):
        n_fft = 2048
        F = np.fft.rfft(f, n=n_fft)
        conv = np.fft.irfft(F * F, n=n_fft) * H
        return conv

    best_val = float('inf')
    best_heights = None
    
    # Initializations
    inits = []
    
    # 1. Box
    inits.append(np.ones(N))
    
    # 2. Semicircle (p=1)
    inits.append((1 - (X / 0.25)**2)**1)
    
    # 3. Super-Gaussian p=2
    inits.append((1 - (X / 0.25)**2)**2)
    
    # 4. Gaussian narrow
    inits.append(np.exp(-((X) / 0.05)**2))
    
    # 5. Gaussian wide
    inits.append(np.exp(-((X) / 0.12)**2))
    
    # 6. Flat-top filter shape (approx)
    inits.append(np.cos(np.pi * X / 0.25)**2)

    for init in inits:
        if time.time() > deadline - 10:
            break
            
        f = init.copy()
        f = np.maximum(f, 0)
        f = normalize(f)
        
        # Iterative optimization
        tau = 0.01 # Temperature for softmax
        eta = 0.05 # Step size
        
        for iter in range(2000):
            if time.time() > deadline - 5:
                break
                
            c = convolve(f)
            # Softmax approximation of max
            # Shift c to avoid overflow
            c_max = np.max(c)
            p = np.exp((c - c_max) / tau)
            p_sum = np.sum(p)
            if p_sum == 0:
                break
            p = p / p_sum
            
            # Gradient G = p * f (convolution)
            # We want to minimize sum p_j c_j
            # d/d f_i (sum p_j c_j) = sum p_j (f_{j-i} + f_{j-i}) = 2 (p * f)_i
            G = convolve(p) # This is p*f convolved? No.
            # Wait. c = f*f.
            # d c_j / d f_i = f_{j-i} + f_{j-i} = 2 f_{j-i}.
            # Sum_j p_j 2 f_{j-i} = 2 (p * f)_i.
            # So G = 2 * convolve_p_f?
            # We need convolution of p and f.
            F_p = np.fft.rfft(p, n=2048)
            F_f = np.fft.rfft(f, n=2048)
            G = 2 * np.fft.irfft(F_p * F_f, n=2048) * H
            
            # Update
            f_new = f - eta * G
            f_new = np.maximum(f_new, 0)
            f_new = normalize(f_new)
            
            # Check for improvement
            val = compute_score(f_new)
            if val < best_val:
                best_val = val
                best_heights = f_new.copy()
            
            f = f_new
            # Decay tau
            tau = max(tau * 0.99, 1e-4)
            # Decay eta
            eta = max(eta * 0.999, 0.001)

    if best_heights is None:
        return [1.0] * N
    
    return list(best_heights)
