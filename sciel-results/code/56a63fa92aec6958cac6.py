import numpy as np
import time
import random
from scipy.optimize import minimize_scalar

N = 400

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit

    def autoconv_max_abs(x):
        """
        Compute max |C[k]| for k != 0, where C[k] = sum_i x[i] x[i+k].
        Uses FFT for speed.
        """
        # Linear convolution via FFT
        n = len(x)
        size = 1
        while size < 2 * n - 1:
            size *= 2
        
        X = np.fft.rfft(x, size)
        C_full = np.fft.irfft(X * X, size)
        
        # C_full has length size, but we only care up to 2n-2
        # C_full[k] for k=0..2n-2
        # Lag 0 is at index 0? No, wait.
        # Convolution of x and x:
        # (x*x)[k] = sum_i x[i] x[k-i]
        # This is not exactly the autocorrelation C[k] = sum_i x[i]x[i+k].
        # Autocorrelation is cross-correlation with x and x.
        # np.correlate(x, x, 'full') gives C[k] where C[k] = sum x[i] x[i+k] for k>=0?
        # Let's stick to np.correlate for correctness or use FFT properly.
        
        # For real signals, autocorrelation C[k] = sum x[i]x[i+k].
        # This is equivalent to the inverse FFT of |FFT(x)|^2, but shifted.
        # Specifically, if we compute r = np.fft.irfft(np.abs(X)**2), 
        # r[0] is C[0] = sum x[i]^2 = N (if x is +/-1).
        # r[1] is C[1] = sum x[i]x[i+1].
        # r[n-1] is C[-(n-1)] = C[n-1].
        
        # So we need max(|r[1]|, ..., |r[n-1]|).
        
        # Let's recompute using the standard FFT method for autocorrelation.
        X = np.fft.rfft(x, size)
        power = np.abs(X)**2
        C = np.fft.irfft(power, size)
        
        # C[0] is lag 0
        # C[1] is lag 1
        # ...
        # C[n-1] is lag n-1
        # C[n] is lag -n+1? No, circular convolution issues if size < 2n-1.
        # Since we padded to size >= 2n-1, the linear autocorrelation is contained in C[0:size-n+1] and C[size-n+1:size]?
        # Actually, for linear autocorrelation of length n signal:
        # C[k] for k=0..n-1 is in C[0..n-1].
        # C[-k] = C[k].
        
        # So we just need max(|C[1:n]|).
        if n == 1:
            return 0.0
        max_abs = np.max(np.abs(C[1:n]))
        return max_abs

    def score(x):
        S = np.sum(x)
        if S == 0:
            return np.inf
        max_c = autoconv_max_abs(x)
        return max_c / (S * S)

    best_x = None
    best_score = np.inf
    
    # 1. Generate diverse random sequences
    # We want low auto-correlation.
    # Random +/-1 sequences have expected max C[k] ~ sqrt(N log N).
    # S^2 ~ N^2.
    # Score ~ sqrt(N log N) / N^2. This is very small.
    # Wait, if x is +/-1, S can be small. If S is small, score is huge.
    # We want S to be large (close to N) and C[k] to be small.
    # But if S is close to N, the sequence is mostly +1.
    # If x is mostly +1, C[1] ~ N, S^2 ~ N^2, score ~ 1/N.
    # If x is random, S ~ sqrt(N), S^2 ~ N. C[k] ~ sqrt(N). Score ~ 1/sqrt(N).
    # 1/N is smaller than 1/sqrt(N) for large N.
    # So we want S to be large.
    # This implies the sequence should have a large DC component.
    
    # Let's try sequences with high bias towards +1, but with some -1s to reduce C[k].
    # Or maybe the optimal solution has a specific structure.
    
    # Let's just do a random search with a twist:
    # Start with all +1.
    # Flip random indices to -1.
    # The score is max |C[k]| / S^2.
    # As we flip more -1s, S decreases.
    # C[k] might decrease too.
    
    # Let's try a few strategies:
    
    # Strategy 1: Pure random +/-1
    for i in range(200):
        if time.time() - start_time > time_limit - 10:
            break
        x = np.random.choice([-1, 1], N)
        s = score(x)
        if s < best_score:
            best_score = s
            best_x = x.copy()
            
    # Strategy 2: High bias +1
    for bias in [0.9, 0.8, 0.7, 0.5]:
        if time.time() - start_time > time_limit - 10:
            break
        for i in range(50):
            x = np.random.choice([-1, 1], N, p=[bias, 1-bias])
            s = score(x)
            if s < best_score:
                best_score = s
                best_x = x.copy()
                
    # Strategy 3: Local search from best
    if best_x is not None:
        x = best_x.copy()
        current_score = best_score
        
        # Simple greedy flip
        # Precompute C? No, just recalc.
        # N=400, FFT is fast.
        
        for _ in range(500):  # Max iterations
            if time.time() - start_time > time_limit - 5:
                break
            
            improved = False
            # Randomly pick a few indices to try
            indices = np.random.permutation(N)
            for i in indices[:20]:
                old_val = x[i]
                x[i] = -old_val
                new_score = score(x)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    x[i] = old_val
            if not improved:
                break
                
        if current_score < best_score:
            best_score = current_score
            best_x = x

    return best_x
