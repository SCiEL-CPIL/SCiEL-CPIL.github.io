"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconvolution_peak(x):
    """Compute the peak absolute value of the autoconvolution."""
    n = len(x)
    # Use next power of 2 >= 2n-1 for efficient FFT
    fft_size = 1 << (2 * n - 1).bit_length()
    X = np.fft.rfft(x, fft_size)
    conv = np.fft.irfft(X * X, fft_size)
    # Linear autoconvolution is the first 2n-1 values
    conv = conv[:2 * n - 1]
    return np.max(np.abs(conv))

def _local_search(x, time_limit=10):
    """Local search: flip individual bits to reduce autoconvolution peak."""
    x = x.copy()
    best_peak = _autoconvolution_peak(x)
    start_time = time.time()
    n = len(x)
    
    indices = np.arange(n)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in indices:
            if (time.time() - start_time) > time_limit:
                break
            # Try flipping position i
            x[i] *= -1
            new_peak = _autoconvolution_peak(x)
            if new_peak < best_peak - 1e-12:
                best_peak = new_peak
                improved = True
            else:
                x[i] *= -1  # revert
    return x, best_peak

def _generate_matern_sequence(n, nu, length_scale, seed):
    """
    Generate a signed step function from a 1D Gaussian Random Field
    with Matérn covariance.
    
    The Matérn kernel is:
    C(r) = (2^(1-nu) / Gamma(nu)) * (sqrt(2*nu) * |r| / length_scale)^nu * K_nu(sqrt(2*nu) * |r| / length_scale)
    
    For 1D, we can sample this by solving the differential equation
    or using spectral methods. However, for simplicity and speed,
    we can approximate the field by summing random sinusoids with
    amplitudes determined by the spectral density of the Matérn field.
    
    The spectral density of the Matérn field in 1D is:
    S(k) = (2*nu)^(2nu) / (Gamma(nu)^2 * (2*nu + k^2 * length_scale^2)^(nu + 0.5))
    
    This is complex to sample directly. An alternative is to use the
    fact that the Matérn field can be generated as a solution to a
    stochastic differential equation, or we can use a simpler approach:
    
    Generate a smooth random field by applying a low-pass filter to
    white noise, where the filter's cutoff frequency is related to
    the length_scale. The smoothness is controlled by the order of
    the filter or the shape of the spectrum.
    
    For the purpose of this task, we will use a simpler heuristic:
    Generate a sequence of Gaussian random variables, then apply a
    moving average filter with a window size related to length_scale.
    The sign of the resulting smooth signal determines the step function.
    
    This is an approximation, but it captures the essence of a smooth
    random field with a characteristic length scale.
    """
    rng = np.random.RandomState(seed)
    
    # Generate white noise
    white_noise = rng.randn(n)
    
    # Apply a moving average filter
    # The window size is related to the length scale
    # If length_scale is small, the field is rough (short window)
    # If length_scale is large, the field is smooth (long window)
    # Let's map length_scale in [1, 10] to window size in [1, 20]
    window_size = int(np.clip(length_scale * 2, 1, 20))
    
    if window_size == 1:
        smoothed = white_noise
    else:
        # Use a simple moving average
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(white_noise, kernel, mode='same')
    
    # Normalize to zero mean and unit variance
    smoothed = (smoothed - np.mean(smoothed)) / (np.std(smoothed) + 1e-12)
    
    # The sign of the smoothed field determines the step function
    signs = np.sign(smoothed)
    signs[signs == 0] = 1.0
    
    return signs

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    best_x = None
    best_peak = np.inf
    
    # Strategy 1: Thue-Morse initialization (known good structure)
    def _thue_morse(n):
        seq = np.zeros(n, dtype=np.int8)
        for i in range(n):
            seq[i] = bin(i).count('1') % 2
        return 2 * seq.astype(np.float64) - 1
    
    x_tm = _thue_morse(N)
    x_refined, peak = _local_search(x_tm, time_limit=20)
    if peak < best_peak:
        best_peak = peak
        best_x = x_refined
    
    # Strategy 2: Gaussian Random Field with Matérn-like smoothing
    # We will search over the 'length_scale' parameter (smoothness)
    # and different random seeds.
    
    # Coarse search over smoothness
    length_scales = np.linspace(1.0, 10.0, 10)
    
    for ls in length_scales:
        if time.time() > deadline - 30:
            break
        
        # Try a few seeds for each smoothness
        for seed in range(5):
            if time.time() > deadline - 20:
                break
            
            x_gf = _generate_matern_sequence(N, 1.5, ls, seed)
            x_refined, peak = _local_search(x_gf, time_limit=10)
            
            if peak < best_peak:
                best_peak = peak
                best_x = x_refined
    
    # Strategy 3: Random restarts with local search
    rng = np.random.RandomState(42)
    for restart in range(5):
        if (time.time() - start_time) > deadline - 10:
            break
        
        x_rand = rng.choice([-1.0, 1.0], size=N)
        x_refined, peak = _local_search(x_rand, time_limit=10)
        
        if peak < best_peak:
            best_peak = peak
            best_x = x_refined
    
    if best_x is None:
        best_x = np.ones(N)
    
    return best_x
# EVOLVE-BLOCK-END
