"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.signal import fftconvolve

def construct():
    # Strategy: Minimize the maximum absolute autocorrelation sidelobe.
    # We assume the metric excludes lag 0 or normalizes by energy, 
    # making low-autocorrelation sequences superior to the constant sequence.
    # We use a local search approach on a random +/- 1 sequence.
    
    np.random.seed(42)
    x = np.random.choice([-1, 1], size=N)
    
    def get_max_sidelobe(seq):
        # Linear autocorrelation
        # R(k) = sum_{i} seq[i] * seq[i+k]
        # Using fftconvolve for speed
        # Reverse sequence for correlation
        corr = fftconvolve(seq, seq[::-1], mode='full')
        # The correlation at lag 0 is at index N-1 in the full output
        # Sidelobes are indices 0 to N-2 and N to 2N-2
        # We want max abs value excluding the peak at N-1
        # Note: fftconvolve output order: lags -(N-1) to N-1
        # Index N-1 corresponds to lag 0.
        # We exclude index N-1.
        peak_idx = N - 1
        max_val = 0
        for i in range(2*N - 1):
            if i != peak_idx:
                val = abs(corr[i])
                if val > max_val:
                    max_val = val
        return max_val

    current_score = get_max_sidelobe(x)
    
    # Simple local search: try flipping random bits
    # N=400, 120s limit. 
    # Each evaluation is O(N log N). 
    # 1000 iterations should be fast.
    for _ in range(1000):
        idx = np.random.randint(N)
        x_new = x.copy()
        x_new[idx] *= -1
        new_score = get_max_sidelobe(x_new)
        if new_score < current_score:
            x = x_new
            current_score = new_score
            
    return x
# EVOLVE-BLOCK-END
