"""Spectral shaping via random band-limited phase masks to minimize C3'."""
import numpy as np
import time

N = 400

def _compute_score(x):
    """Compute max|autoconv| / (sum x)^2."""
    s = np.sum(x)
    if s == 0:
        return np.inf
    # Full linear autocorrelation
    # For sequence x of length N, autoconv at lag k: sum_i x[i] * x[i+k] (with zero padding)
    # Use FFT for efficiency
    n_fft = 2 * N
    X = np.fft.rfft(x, n=n_fft)
    C = np.fft.irfft(X * np.conj(X), n=n_fft)
    # C[0] is sum x^2, C[k] for k>0 is autocorrelation at positive lag
    # We want max over all lags of |C[k]|
    # The autoconvolution here is the linear autocorrelation
    # Note: C[k] for k=0..N-1 are the positive lags
    # We need max |C[k]| over all k
    max_abs = np.max(np.abs(C[:N]))
    return max_abs / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_score = np.inf
    best_x = np.ones(N)
    
    # Initial: Rudin-Shapiro-like or simple constructions
    # Try a few deterministic starting points
    
    # 1. All ones (baseline)
    x0 = np.ones(N)
    s0 = _compute_score(x0)
    if s0 < best_score:
        best_score = s0
        best_x = x0.copy()
    
    # 2. Rudin-Shapiro sequence
    def rudin_shapiro(n):
        """Generate Rudin-Shapiro sequence of length n."""
        rs = np.array([1, 1])
        while len(rs) < n:
            rs = np.concatenate([rs, rs * np.array([1, -1])])
        return rs[:n]
    
    x1 = rudin_shapiro(N).astype(float)
    s1 = _compute_score(x1)
    if s1 < best_score:
        best_score = s1
        best_x = x1.copy()
    
    # 3. Random initializations via spectral shaping
    # The idea: generate random phase masks in frequency domain, 
    # take inverse FFT, get real-valued low-peak signals
    
    np.random.seed(42)
    
    # Spectral shaping: random phase with unit amplitude in a band
    # This creates signals with controlled peak-to-average ratio
    
    num_trials = 0
    band_widths = [0.3, 0.5, 0.7, 1.0]  # Fraction of frequency band to use
    
    for bw in band_widths:
        while time.time() < deadline:
            # Generate random phase mask
            # Use only positive frequencies for real signal
            n_freq = N // 2 + 1
            
            # Random phases
            phases = np.random.uniform(0, 2 * np.pi, size=n_freq)
            
            # Apply band-limiting: only use a fraction of frequencies
            # Keep DC and a random band of frequencies
            n_band = max(1, int(n_freq * bw))
            
            # Create amplitude mask: 1 in band, 0 elsewhere
            amp = np.zeros(n_freq)
            amp[0] = 1  # Keep DC
            # Choose random subset of frequencies
            freqs = np.arange(1, n_freq)
            selected = np.random.choice(freqs, size=min(n_band - 1, len(freqs)), replace=False)
            amp[selected] = 1
            
            # Construct spectrum with random phases and band-limited amplitudes
            spectrum = amp * np.exp(1j * phases)
            
            # Inverse FFT to get time-domain signal
            x = np.fft.irfft(spectrum, n=N)
            
            # Ensure real and reasonable
            x = np.real(x)
            
            # Normalize to have sum = 1 for fair comparison
            s = np.sum(x)
            if abs(s) > 1e-10:
                x = x / s
            
            score = _compute_score(x)
            num_trials += 1
            
            if score < best_score:
                best_score = score
                best_x = x.copy()
            
            # Check time
            if time.time() > deadline - 5.0:
                break
    
    # Local refinement: try flipping signs of individual elements
    # Simple coordinate descent on sign flips
    if time.time() < deadline - 10.0:
        x_current = best_x.copy()
        current_score = best_score
        improved = True
        max_iters = 1000
        
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            if not improved:
                break
            improved = False
            
            # Try flipping a random subset of signs
            for _ in range(50):
                if time.time() > deadline - 5.0:
                    break
                
                # Pick a random index to flip
                idx = np.random.randint(N)
                x_trial = x_current.copy()
                x_trial[idx] *= -1
                
                s_trial = np.sum(x_trial)
                if abs(s_trial) < 1e-10:
                    continue
                x_trial = x_trial / s_trial
                
                score_trial = _compute_score(x_trial)
                
                if score_trial < current_score - 1e-12:
                    x_current = x_trial
                    current_score = score_trial
                    improved = True
    
    if best_score < np.inf:
        return best_x
    return best_x

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
