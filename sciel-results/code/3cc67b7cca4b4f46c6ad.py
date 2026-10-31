"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 600

# EVOLVE-BLOCK-START
def construct():
    """
    Minimize the peak autocorrelation sidelobe of a nonnegative step function
    with N equal intervals on [-1/4, 1/4].
    
    Tries several candidate window functions and picks the one with lowest
    peak sidelobe. Then applies iterative refinement.
    """
    start_time = time.time()
    deadline = 140.0
    
    def compute_peak_sidelobe(h):
        """Compute peak autocorrelation sidelobe ratio."""
        norm = np.sqrt(np.sum(h**2))
        if norm < 1e-15:
            return 1.0
        h_norm = h / norm
        n_fft = 2 * N
        H = np.fft.rfft(h_norm, n=n_fft)
        autocorr = np.fft.irfft(np.abs(H)**2, n=n_fft)
        pos_lags = autocorr[1:N]
        neg_lags = autocorr[n_fft - N + 1:n_fft]
        return max(np.max(pos_lags), np.max(neg_lags))
    
    # Generate candidate windows
    candidates = []
    
    # Uniform (baseline)
    candidates.append(np.ones(N))
    
    # Raised cosine
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    candidates.append(0.5 * (1 + np.cos(x)))
    
    # Hamming
    candidates.append(0.54 - 0.46 * np.cos(2*np.pi*np.arange(N)/N))
    
    # Hanning
    candidates.append(0.5 * (1 - np.cos(2*np.pi*np.arange(N)/N)))
    
    # Blackman
    n = np.arange(N)
    blackman = (0.42 - 0.5*np.cos(2*np.pi*n/N) + 0.08*np.cos(4*np.pi*n/N))
    candidates.append(np.maximum(blackman, 0))
    
    # Evaluate all candidates
    best_peak = np.inf
    best_h = None
    for h in candidates:
        h = np.maximum(h, 0)  # ensure nonnegative
        if np.sum(h**2) < 1e-15:
            continue
        peak = compute_peak_sidelobe(h)
        if peak < best_peak:
            best_peak = peak
            best_h = h.copy()
    
    if best_h is None:
        return np.ones(N).tolist()
    
    # Iterative refinement: reduce peak sidelobe
    h = best_h.copy()
    best_h_refined = h.copy()
    best_peak_refined = best_peak
    
    for iteration in range(100):
        if time.time() - start_time > deadline:
            break
        
        h_norm = h / np.sqrt(np.sum(h**2))
        n_fft = 2 * N
        H = np.fft.rfft(h_norm, n=n_fft)
        autocorr = np.fft.irfft(np.abs(H)**2, n=n_fft)
        pos_lags = autocorr[1:N]
        neg_lags = autocorr[n_fft - N + 1:n_fft]
        
        peak_pos = np.max(pos_lags)
        peak_neg = np.max(neg_lags)
        
        if peak_pos >= peak_neg:
            lag = np.argmax(pos_lags) + 1
            # Compute contribution to this lag
            contrib = np.zeros(N)
            for i in range(N - lag):
                contrib[i] += h_norm[i + lag]
                contrib[i + lag] += h_norm[i]
        else:
            lag = np.argmax(neg_lags) + 1
            contrib = np.zeros(N)
            for i in range(lag, N):
                contrib[i] += h_norm[i - lag]
                contrib[i - lag] += h_norm[i]
        
        # Reduce heights where contribution is high
        max_c = np.max(contrib)
        if max_c < 1e-10:
            break
        h = h - 0.005 * contrib / max_c
        h = np.maximum(h, 0)
        
        norm_sq = np.sum(h**2)
        if norm_sq < 1e-15:
            break
        h = h / np.sqrt(norm_sq)
        
        peak = compute_peak_sidelobe(h)
        if peak < best_peak_refined:
            best_peak_refined = peak
            best_h_refined = h.copy()
    
    return best_h_refined.tolist()
# EVOLVE-BLOCK-END
