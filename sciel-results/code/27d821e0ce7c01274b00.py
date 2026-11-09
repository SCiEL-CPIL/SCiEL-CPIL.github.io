"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autocorr_peak(x):
    """Compute the maximum absolute value of the periodic autocorrelation sidelobes."""
    n = len(x)
    # Use FFT for fast autocorrelation
    f = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    # For the C3 bound, we care about the max off-peak autocorrelation
    peak = np.max(np.abs(acf[1:]))
    return peak


def _bandpass_init(bandwidth=0.3, seed=42):
    """Initialize with a smooth low-frequency signal via bandpass filtering in Fourier domain."""
    rng = np.random.RandomState(seed)
    n = N
    
    # Start with random signs
    x = np.where(rng.rand(n) < 0.5, 1.0, -1.0)
    
    # Apply bandpass filter in Fourier domain to suppress high frequencies
    F = np.fft.fft(x)
    
    # Create a bandpass mask: keep low frequencies, attenuate high frequencies
    freqs = np.fft.fftfreq(n)
    # Smooth bandpass: Gaussian window centered at zero frequency
    sigma = bandwidth * n / 2.0
    mask = np.exp(-0.5 * (freqs / sigma) ** 2)
    
    # Apply filter
    F_filtered = F * mask
    x_filtered = np.fft.ifft(F_filtered).real
    
    # Quantize back to ±1 based on sign
    x_out = np.where(x_filtered >= 0, 1.0, -1.0)
    
    return x_out


def _local_search(x, max_time=100.0):
    """Local search to minimize peak autocorrelation."""
    start_time = time.time()
    best_x = x.copy()
    best_peak = _autocorr_peak(x)
    n = len(x)
    
    # Multi-pass local search
    while time.time() - start_time < max_time:
        improved = False
        for idx in range(n):
            if time.time() - start_time > max_time:
                break
            x[idx] = -x[idx]
            new_peak = _autocorr_peak(x)
            if new_peak < best_peak - 1e-12:
                best_peak = new_peak
                best_x = x.copy()
                improved = True
            else:
                x[idx] = -x[idx]  # revert
        if not improved:
            # Try random restarts of local search from perturbed states
            if time.time() - start_time > max_time - 10:
                break
            # Perturb: flip 5 random bits
            for _ in range(5):
                idx = np.random.randint(n)
                x[idx] = -x[idx]
            x_peak = _autocorr_peak(x)
            if x_peak < best_peak:
                best_peak = x_peak
                best_x = x.copy()
    
    return best_x, best_peak


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initialize with bandpass-filtered smooth signal
    np.random.seed(42)
    x = _bandpass_init(bandwidth=0.3, seed=42)
    
    # Initial peak
    current_peak = _autocorr_peak(x)
    best_x = x.copy()
    best_peak = current_peak
    
    # Local search
    best_x, best_peak = _local_search(x, max_time=100.0)
    
    # Check deadline
    if time.time() > deadline:
        return best_x
    
    # Try a few more local searches from different bandpass initializations
    for seed in range(1, 20):
        if time.time() > deadline - 5.0:
            break
        np.random.seed(seed)
        x2 = _bandpass_init(bandwidth=0.3, seed=seed)
        # Small perturbation
        for _ in range(3):
            idx = np.random.randint(N)
            x2[idx] = -x2[idx]
        
        x2, peak2 = _local_search(x2, max_time=min(10.0, deadline - time.time() - 2.0))
        
        if peak2 < best_peak - 1e-12:
            best_peak = peak2
            best_x = x2
    
    return best_x
# EVOLVE-BLOCK-END
