"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _autocorr_peak(h):
    """Compute max |autocorrelation| at non-zero lags using FFT."""
    n = len(h)
    if n <= 1:
        return 0.0
    # Zero-pad to 2*n for linear autocorrelation
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    # acf[0] is lag 0 (sum of squares), we want max |acf[k]| for k=1..n-1
    # Also consider negative lags: acf[n+1] to acf[fft_size-1] correspond to negative lags
    # But for real sequences, acf is symmetric, so max over 1..n-1 suffices
    return float(np.max(np.abs(acf[1:n])))


def _compute_c3_bound(h):
    """Compute the upper bound on C3 (third autocorrelation inequality).
    
    For a signed step function, the C3 bound involves the maximum of the
    autoconvolution. We use the L-infinity norm of the autocorrelation
    at non-zero lags as the metric to minimize.
    """
    return _autocorr_peak(h)


def _wavelet_based_init(seed=42):
    """Generate initial ±1 sequence using wavelet-based construction.
    
    Uses a multi-resolution approach: construct the sequence as the sign
    of a sum of wavelet-like basis functions with optimized coefficients.
    """
    rng = np.random.RandomState(seed)
    n = N
    
    # Create a multi-scale basis using Haar-like wavelets at different scales
    # We'll use random Gaussian windows at different scales as our "wavelets"
    num_scales = 6
    scales = np.array([2, 4, 8, 16, 32, 64])  # Widths of Gaussian windows
    
    # Random coefficients for each scale
    coeffs = rng.randn(num_scales) * 0.5
    
    # Build the signal as a sum of modulated Gaussian windows
    t = np.arange(n)
    signal = np.zeros(n)
    
    for scale_idx, scale in enumerate(scales):
        # Create a Gaussian window
        center = rng.randint(0, n)
        width = max(1, int(scale))
        gaussian = np.exp(-((t - center) ** 2) / (2 * width ** 2))
        # Modulate with a random phase
        phase = rng.rand() * 2 * np.pi
        modulation = np.cos(2 * np.pi * t / n * (scale_idx + 1) + phase)
        signal += coeffs[scale_idx] * gaussian * modulation
    
    # Normalize and threshold
    signal = signal - np.mean(signal)
    std = np.std(signal)
    if std > 1e-10:
        signal = signal / std
    
    h = np.sign(signal)
    h[h == 0] = 1.0
    
    return h


def _gradient_descent_step(h, lr=0.01):
    """Perform a single gradient descent step on the continuous relaxation.
    
    Instead of binary flips, we relax the problem to continuous values in [-1, 1]
    and use gradient information to guide the search.
    """
    n = len(h)
    h_cont = h.astype(np.float64)
    
    # Compute gradient of the autocorrelation peak with respect to h
    # The autocorrelation at lag k is sum_i h[i] * h[i+k]
    # The gradient with respect to h[i] involves all lags where i is involved
    
    # For efficiency, we compute the gradient using FFT-based methods
    # The peak is at some lag k*, so we focus on that lag
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    
    H = np.fft.rfft(h_cont, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    
    # Find the lag with maximum |acf|
    peak_lag = np.argmax(np.abs(acf[1:n])) + 1
    peak_val = acf[peak_lag]
    
    # Gradient of acf[peak_lag] with respect to h[i]
    # acf[k] = sum_i h[i] * h[i+k] (for k >= 0)
    # d(acf[k])/d(h[i]) = h[i+k] + h[i-k] (with boundary handling)
    
    grad = np.zeros(n)
    for i in range(n):
        # Contribution from h[i] * h[i+k]
        if i + peak_lag < n:
            grad[i] += h_cont[i + peak_lag]
        # Contribution from h[i] * h[i-k] (i.e., h[i-k] * h[i])
        if i - peak_lag >= 0:
            grad[i] += h_cont[i - peak_lag]
    
    # Sign of gradient determines direction
    sign_grad = np.sign(grad)
    # Update: move against the gradient to reduce the peak
    h_new = h_cont - lr * sign_grad
    # Project back to [-1, 1]
    h_new = np.clip(h_new, -1.0, 1.0)
    
    # Threshold to ±1
    h_binary = np.sign(h_new)
    h_binary[h_binary == 0] = 1.0
    
    return h_binary


def _simulated_annealing(h, max_time, T_start=1.0, T_end=0.01):
    """Simulated annealing to escape local minima.
    
    Uses the autocorrelation peak as the energy function and allows
    uphill moves with probability exp(-delta/T).
    """
    n = len(h)
    h = h.copy()
    current_peak = _autocorr_peak(h)
    start_time = time.time()
    
    # Number of iterations to run
    n_iter = int((max_time - (time.time() - start_time)) * 1000)
    n_iter = max(1000, min(n_iter, 100000))
    
    for iter_idx in range(n_iter):
        if time.time() - start_time > max_time:
            break
        
        # Temperature schedule
        progress = iter_idx / n_iter
        T = T_start * (T_end / T_start) ** progress
        
        # Random flip
        i = np.random.randint(0, n)
        old_val = h[i]
        h[i] = -h[i]
        new_peak = _autocorr_peak(h)
        
        delta = new_peak - current_peak
        
        if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-10)):
            current_peak = new_peak
        else:
            h[i] = old_val
    
    return h, current_peak


def construct():
    """Construct a signed step function with N=400 intervals that minimizes
    the autoconvolution upper bound using wavelet-based initialization and
    simulated annealing."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    best_h = None
    best_peak = float('inf')
    
    # Try multiple wavelet-based initializations with different seeds
    seeds = [42, 123, 456, 789, 1000, 2024]
    
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        
        # Generate wavelet-based initial sequence
        h_init = _wavelet_based_init(seed)
        initial_peak = _autocorr_peak(h_init)
        
        if initial_peak < best_peak:
            best_peak = initial_peak
            best_h = h_init.copy()
        
        # Run simulated annealing with remaining time budget
        remaining = deadline - time.time() - 5
        if remaining < 2:
            break
        
        # Split time between SA and local search
        sa_time = remaining * 0.7
        ls_time = remaining * 0.3
        
        # Simulated annealing
        h_sa, peak_sa = _simulated_annealing(h_init, sa_time)
        
        if peak_sa < best_peak:
            best_peak = peak_sa
            best_h = h_sa.copy()
        
        # Local search refinement
        if time.time() < deadline - 2:
            h_ls, peak_ls = _local_search(h_sa, ls_time)
            
            if peak_ls < best_peak:
                best_peak = peak_ls
                best_h = h_ls.copy()
    
    # If no good result, fall back to random initialization
    if best_h is None:
        np.random.seed(42)
        best_h = np.random.choice([-1.0, 1.0], size=N)
    
    return best_h


def _local_search(h, max_time):
    """Greedy local search: flip bits to reduce autocorrelation peak."""
    n = len(h)
    h = h.copy()
    current_peak = _autocorr_peak(h)
    start_time = time.time()
    
    # Precompute positions
    positions = np.arange(n)
    
    while time.time() - start_time < max_time:
        improved = False
        # Shuffle positions to avoid systematic bias
        np.random.shuffle(positions)
        
        for i in positions:
            if time.time() - start_time > max_time:
                break
            # Try flipping position i
            old_val = h[i]
            h[i] = -h[i]
            new_peak = _autocorr_peak(h)
            if new_peak < current_peak:
                current_peak = new_peak
                improved = True
            else:
                h[i] = old_val
        
        if not improved:
            # Try flipping pairs of positions
            for _ in range(50):
                if time.time() - start_time > max_time:
                    break
                i, j = np.random.choice(n, 2, replace=False)
                h[i] = -h[i]
                h[j] = -h[j]
                new_peak = _autocorr_peak(h)
                if new_peak < current_peak:
                    current_peak = new_peak
                    improved = True
                else:
                    h[i] = -h[i]
                    h[j] = -h[j]
            
            if not improved:
                break
    
    return h, current_peak
# EVOLVE-BLOCK-END
