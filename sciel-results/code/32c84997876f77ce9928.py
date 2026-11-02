"""Optimized program for the third autocorrelation inequality task."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def _compute_peak_sidelobe(s):
    """Compute the peak absolute sidelobe of the autocorrelation of binary sequence s."""
    # Compute full autocorrelation via FFT for speed
    n_fft = 1 << (2 * N - 1).bit_length()
    S = np.fft.rfft(s, n=n_fft)
    acf = np.fft.irfft(S * np.conj(S), n=n_fft)[:N]
    # acf[0] is the main lobe (= N for ±1 sequence), sidelobes are acf[1:]
    # Also need negative lags: acf_neg = acf[1:][::-1]
    # Peak sidelobe is max of |acf[1:]| and |acf[1:][::-1]|, but since acf is symmetric,
    # peak_sidelobe = max(|acf[1:]|)
    return np.max(np.abs(acf[1:]))


def _compute_acf_full(s):
    """Compute the full autocorrelation (both positive and negative lags)."""
    n_fft = 1 << (2 * N - 1).bit_length()
    S = np.fft.rfft(s, n=n_fft)
    acf = np.fft.irfft(S * np.conj(S), n=n_fft)
    # acf[0] = R[0], acf[1:N] = R[1],...,R[N-1], acf[N+1:] = R[-1],...,R[-(N-1)]
    # Actually for real input, acf is symmetric: acf[k] = acf[n_fft-k]
    # The positive lags are acf[1:N], negative lags are acf[N+1:2*N] (reversed)
    return acf


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Initialize with random ±1 sequence
    best_s = None
    best_peak = np.inf
    
    # Try a few initial configurations
    rng = np.random.default_rng(42)
    
    # Start with random initialization
    s = rng.choice([-1.0, 1.0], size=N)
    
    # Compute initial peak sidelobe
    peak = _compute_peak_sidelobe(s)
    best_s = s.copy()
    best_peak = peak
    
    # Simulated annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    steps_per_temp = 50
    
    current_peak = peak
    current_s = s.copy()
    
    # Precompute FFT size
    n_fft = 1 << (2 * N - 1).bit_length()
    
    while T > T_min:
        for _ in range(steps_per_temp):
            if time.time() > deadline:
                break
            
            # Randomly flip a block of consecutive elements
            block_len = rng.integers(1, 11)  # block size 1-10
            pos = rng.integers(0, N - block_len + 1)
            
            # Save old values
            old_vals = current_s[pos:pos + block_len].copy()
            old_peak = current_peak
            
            # Flip the block
            current_s[pos:pos + block_len] *= -1
            
            # Compute new peak sidelobe
            new_peak = _compute_peak_sidelobe(current_s)
            
            # Metropolis acceptance
            delta = new_peak - old_peak
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                # Accept
                current_peak = new_peak
                if new_peak < best_peak:
                    best_peak = new_peak
                    best_s = current_s.copy()
            else:
                # Reject: undo
                current_s[pos:pos + block_len] = old_vals
        
        T *= cooling
    
    # Return the best sequence found
    return best_s
# EVOLVE-BLOCK-END
