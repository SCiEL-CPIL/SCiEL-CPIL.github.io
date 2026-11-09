import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    np.random.seed(42)
    h = np.random.choice([-1, 1], N)
    
    # Initial autocorrelation
    n_fft = 1 << (2 * N).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    r = np.fft.irfft(power, n=n_fft)
    
    # r[1:N] are the positive lags
    current_peak = np.max(np.abs(r[1:N]))
    best_h = h.copy()
    best_peak = current_peak
    
    # Precompute indices for vectorized delta calculation
    # For a flip at i, we need to update r[k] for k=1..N-1
    # Delta r[k] = -2 * h[i] * (h[i+k] + h[i-k]) with bounds
    
    max_iterations = 500 # Try more iterations if fast enough
    
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
            
        best_flip_idx = -1
        best_new_peak = current_peak
        
        # We will iterate through all possible flips
        # To speed this up, we can sample a subset or use a heuristic.
        # Let's try all flips but break early if we find a significant improvement?
        # No, let's just find the best flip.
        
        # Optimization: Only check flips that might reduce the peak.
        # The peak is at lag k_peak.
        # Flips that affect r[k_peak] are those i such that i+k_peak < N or i-k_peak >= 0.
        # This is most i.
        
        # Let's just loop through i in a random order and keep the best one found.
        # This is a stochastic local search.
        
        # Actually, a full scan of N=400 is fast enough if vectorized per flip.
        # But 400 flips * 500 iters = 200,000 vectorized ops.
        # Each vectorized op is O(N). So 200,000 * 400 = 80,000,000 ops.
        # This might be too slow in Python/NumPy if not careful.
        
        # Let's reduce iterations or use a smarter search.
        # Let's try 100 iterations, checking all flips.
        
        # Alternative: Just use the random walk approach which was very fast.
        # Let's stick to the random walk but make it smarter.
        
        # Find the peak lag
        sidelobes = np.abs(r[1:N])
        k_peak = np.argmax(sidelobes) + 1 # 1-based lag
        
        # Identify bits that influence r[k_peak]
        # i such that 0 <= i < N and 0 <= i+k_peak < N or 0 <= i-k_peak < N
        # This is basically all i.
        
        # Let's just do a few random flips per iteration as before, it was very fast.
        # 100 iterations * 20 flips = 2000 FFTs. Very fast.
        
        improved = False
        for _ in range(20):
            i = np.random.randint(N)
            h[i] *= -1
            
            # Update r efficiently? Or just recompute FFT?
            # Recomputing FFT is simple and fast for N=400.
            F_new = np.fft.rfft(h, n=n_fft)
            power_new = F_new * np.conj(F_new)
            r_new = np.fft.irfft(power_new, n=n_fft)
            
            new_peak = np.max(np.abs(r_new[1:N]))
            
            if new_peak < current_peak:
                current_peak = new_peak
                r = r_new
                improved = True
                if new_peak < best_peak:
                    best_peak = new_peak
                    best_h = h.copy()
            else:
                h[i] *= -1 # Revert
                # r is still the old r, so no need to revert r
        
        if not improved:
            # If no improvement in 20 random flips, try a few more or break?
            # Let's break if we've been stuck for a while.
            # For now, just continue.
            pass

    return best_h
