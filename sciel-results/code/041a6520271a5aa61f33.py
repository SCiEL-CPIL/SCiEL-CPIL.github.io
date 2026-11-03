import numpy as np
import time

N = 600

def _autoconv_max(step_heights):
    """Compute the maximum of the autoconvolution of the step function."""
    # The step function has N equal intervals on [-1/4, 1/4]
    # Autoconvolution: (f * f)(t) = integral f(t-s) f(s) ds
    # For step functions, we can compute via FFT
    # Pad to avoid circular convolution artifacts
    n_fft = 2 * N
    f_padded = np.zeros(n_fft)
    f_padded[:N] = step_heights
    # FFT
    F = np.fft.rfft(f_padded)
    # Autoconvolution in frequency domain is |F|^2
    ac = np.fft.irfft(np.abs(F)**2, n_fft)
    # The autoconvolution is supported on [-1/2, 1/2] approximately
    # We want the maximum over the relevant range
    # The step function is on [-1/4, 1/4], so autoconv is on [-1/2, 1/2]
    # In our discretization with n_fft points covering [-1/2, 1/2] (circular)
    # Actually, let's think more carefully.
    # The intervals are on [-1/4, 1/4], each of width 1/(4*N)
    # Autoconvolution of two functions supported on [-1/4, 1/4] is supported on [-1/2, 1/2]
    # With n_fft = 2*N points, the spacing is 1/(2*N) which matches the interval width
    # The result ac has n_fft/2 + 1 real values from irfft
    # We need to find the max of the autoconvolution
    return np.max(ac)

def _normalize_autoconv_max(step_heights):
    """Normalized maximum of autoconvolution - this is the upper bound on C1."""
    # The normalization: for a step function with heights h_i on intervals of width w
    # The autoconvolution at t=0 is sum h_i^2 * w
    # The "normalized" max likely means max(autoconv) / (integral f)^2 or similar
    # Let's compute: max of autoconv divided by (sum of h_i * w)^2 where w = 1/(4*N)
    w = 1.0 / (4.0 * N)
    integral_f = np.sum(step_heights) * w
    if integral_f == 0:
        return 0.0
    ac_max = _autoconv_max(step_heights)
    # The autoconvolution value includes the width factor
    # Let's just return ac_max / (integral_f ** 2) as a normalized measure
    return ac_max / (integral_f ** 2)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # 10 seconds margin before 150s kill
    
    # Initialize with a smooth positive function in frequency domain
    # Use a Gaussian-like spectrum to start
    n_freq = N // 2 + 1  # number of real FFT bins
    freqs = np.fft.rfftfreq(n_freq, d=1.0/N)
    
    # Start with a flat spectrum (uniform time domain)
    # But let's try a slightly shaped initial spectrum
    init_mag = np.ones(n_freq)
    
    best_score = float('inf')
    best_heights = np.ones(N)
    
    # First compute baseline
    baseline_score = _normalize_autoconv_max(best_heights)
    best_score = baseline_score
    
    # Coordinate descent on frequency magnitudes
    # We'll optimize the magnitude at each frequency bin
    # The time-domain signal must be nonnegative
    
    # Use softplus projection: given a real signal x, project to nonnegative
    # by taking softplus(x) or just clip(x, 0)
    # For smoother optimization, we can work with log-magnitudes or use a different parameterization
    
    # Let's try: represent f via its FFT, optimize the real and imaginary parts
    # with constraint that the inverse FFT is nonnegative
    
    # Simpler approach: coordinate descent on time-domain heights with nonnegativity
    # But the knowledge says spectral approach works better
    
    # Let's do: iterate on frequency magnitudes, project to time domain, clip to nonnegative
    # Then recompute FFT
    
    current_mag = init_mag.copy()
    current_phase = np.zeros(n_freq)
    
    # Iterative optimization
    max_iters = 5000
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Randomly select a frequency bin to optimize
        idx = np.random.randint(1, n_freq - 1)  # skip DC and Nyquist
        
        # Try different magnitudes at this bin
        # Use a small step
        best_local_score = best_score
        best_local_mag = current_mag[idx]
        
        # Try a few candidate values
        for delta in [-0.1, -0.05, 0.05, 0.1]:
            candidate_mag = current_mag[idx] + delta
            if candidate_mag < 0:
                candidate_mag = 0.0
            
            # Build full spectrum
            F = np.zeros(n_freq, dtype=complex)
            F[:] = current_mag * np.exp(1j * current_phase)
            F[idx] = candidate_mag * np.exp(1j * current_phase[idx])
            
            # Inverse FFT to get time domain
            f_time = np.fft.irfft(F, N)
            
            # Project to nonnegative
            f_time = np.maximum(f_time, 0)
            
            # Compute score
            score = _normalize_autoconv_max(f_time)
            
            if score < best_local_score:
                best_local_score = score
                best_local_mag = candidate_mag
        
        # Update
        if best_local_mag != current_mag[idx]:
            current_mag[idx] = best_local_mag
            if best_local_score < best_score:
                best_score = best_local_score
                # Update best_heights
                F = current_mag * np.exp(1j * current_phase)
                f_time = np.fft.irfft(F, N)
                best_heights = np.maximum(f_time, 0)
    
    # Final projection
    F = current_mag * np.exp(1j * current_phase)
    f_time = np.fft.irfft(F, N)
    result = np.maximum(f_time, 0)
    
    # Ensure nonnegative
    result = np.maximum(result, 0)
    
    return list(result)
# EVOLVE-BLOCK-END
