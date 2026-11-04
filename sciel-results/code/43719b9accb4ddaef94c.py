import numpy as np
from scipy.signal import fftconvolve

N = 400

def construct():
    # Target size for the support.
    # We want a sparse vector to minimize the peak of the autoconvolution.
    # A Sidon set provides a good structure, but we can optimize signs.
    M = 20
    
    # We will perform a local search to minimize the max absolute autocorrelation
    # (excluding lag 0) for a vector with support of size M.
    
    # 1. Initialize a random support
    # Using a random permutation ensures distinct indices
    support = np.random.choice(N, M, replace=False)
    
    # 2. Initialize random signs
    signs = np.random.choice([-1, 1], M)
    
    # Helper to compute the vector
    def get_vector(support, signs):
        x = np.zeros(N)
        x[support] = signs / np.sqrt(M)
        return x

    # Helper to compute the peak autocorrelation (circular, non-zero lags)
    def peak_ac(x):
        # Circular autocorrelation
        # Use FFT for efficiency
        X = np.fft.fft(x)
        ac = np.fft.ifft(X * np.conj(X)).real
        # Normalize? The vector is L2=1, so ac[0] = 1.
        # We want to minimize max(ac[1:], ac[:N-1]) in absolute value.
        # ac is real and symmetric.
        # Exclude lag 0.
        return np.max(np.abs(ac[1:]))

    # Local search
    current_x = get_vector(support, signs)
    current_peak = peak_ac(current_x)
    
    # Try to improve by flipping signs
    # We do a few rounds of sign flipping
    for _ in range(100):
        improved = False
        for i in range(M):
            # Flip sign i
            signs[i] *= -1
            new_x = get_vector(support, signs)
            new_peak = peak_ac(new_x)
            if new_peak < current_peak:
                current_peak = new_peak
                current_x = new_x
                improved = True
            else:
                # Revert
                signs[i] *= -1
        
        if not improved:
            break
            
    # Optional: Try to adjust support (swap an index)
    # This is more expensive but might find a better structure.
    # Given the 120s limit, we can do a few iterations.
    for _ in range(20):
        improved = False
        for i in range(M):
            for j in range(N):
                if j not in support:
                    # Swap support[i] with j
                    old_val = support[i]
                    support[i] = j
                    new_x = get_vector(support, signs)
                    new_peak = peak_ac(new_x)
                    if new_peak < current_peak:
                        current_peak = new_peak
                        current_x = new_x
                        improved = True
                        break # Try next i
                    else:
                        support[i] = old_val
        if not improved:
            break

    return current_x
