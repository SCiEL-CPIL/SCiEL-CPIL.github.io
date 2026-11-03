"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

New approach: Peak-shaving with simulated annealing and spectral initialization.
We start from a smooth parabolic/radial basis function and iteratively reduce
the peak of the autoconvolution by adjusting bins based on their contribution
to the peak lags, using a cooling schedule for step sizes.
"""

import numpy as np
import time

N = 600
TIME_LIMIT = 115  # safety margin before 120s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    conv = np.convolve(f, f, mode='full')
    return np.max(conv) / (s * s)

def initial_guess_parabolic():
    """Create a parabolic (radial basis) initial guess."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    f = np.maximum(1.0 - (x / 0.25) ** 2, 0.0)
    return f

def initial_guess_cosine():
    """Create a cosine bump initial guess."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    f = 0.5 * (1.0 + np.cos(2 * np.pi * x))
    return f

def peak_shaving(f, start_time, max_time, rng=None):
    """Iteratively reduce the peak of the autoconvolution.
    
    At each step:
    1. Compute autoconvolution and find peak lags
    2. For each bin j, compute sensitivity: how much it contributes to peak lags
    3. Reduce bins with high sensitivity, increase bins with low sensitivity
    4. Normalize and check if improvement occurred
    """
    if rng is None:
        rng = np.random.RandomState(42)
    
    best_f = f.copy()
    best_val = compute_score(f)
    current_f = f.copy()
    
    # Cooling schedule
    T = 0.1  # initial "temperature" (step size)
    cooling = 0.995
    
    for iteration in range(50000):
        if time.time() - start_time > max_time:
            break
        
        s = np.sum(current_f)
        if s < 1e-12:
            break
        # Normalize to sum = N
        current_f = current_f * (N / s)
        
        conv = np.convolve(current_f, current_f, mode='full')
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_f = current_f.copy()
        
        # Find peak lags (within top fraction)
        threshold = max_c * 0.98
        peak_indices = np.where(conv >= threshold)[0]
        
        if len(peak_indices) == 0:
            break
        
        # Compute sensitivity for each bin
        # conv[m] = sum_j f[j] * f[m-j]
        # d(conv[m])/d(f[j]) = f[m-j] (for valid m-j)
        sensitivity = np.zeros(N)
        for m in peak_indices:
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            for j in range(j_min, j_max + 1):
                sensitivity[j] += current_f[m - j]
        
        # Normalize sensitivity
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        # Create update: reduce high-sensitivity bins, increase low-sensitivity
        # Add noise for simulated annealing effect
        noise = rng.normal(0, 0.1, size=N)
        update = -T * sens_norm + T * 0.5 * noise
        
        # Apply update
        f_new = current_f * (1.0 + update)
        f_new = np.maximum(f_new, 0.0)
        
        s_new = np.sum(f_new)
        if s_new < 1e-12:
            T *= cooling
            continue
        f_new = f_new * (N / s_new)
        
        conv_new = np.convolve(f_new, f_new, mode='full')
        val_new = np.max(conv_new) / (N * N)
        
        # Accept if better or with probability exp(-delta/T)
        if val_new < current_val:
            current_f = f_new
            T *= cooling
        else:
            delta = val_new - current_val
            if delta > 0:
                accept_prob = np.exp(-delta / (T + 1e-12))
                if rng.random() < accept_prob:
                    current_f = f_new
            T *= cooling
        
        # Early stopping if temperature too low
        if T < 1e-6:
            break
    
    return best_f, best_val

def spectral_refinement(f, start_time, max_time):
    """Refine using frequency domain insights.
    
    The autoconvolution peak is related to the L-infinity norm of the inverse
    Fourier transform of |F|^2. We can try to flatten the power spectrum.
    """
    best_f = f.copy()
    best_val = compute_score(f)
    current_f = f.copy()
    
    # Get FFT of current function
    F = np.fft.rfft(current_f, n=2*N)
    power = np.abs(F) ** 2
    
    # We want to reduce the peak of autoconvolution, which corresponds to
    # reducing the high-frequency components that cause sharp peaks
    # Try to smooth the power spectrum
    for iteration in range(1000):
        if time.time() - start_time > max_time:
            break
        
        s = np.sum(current_f)
        if s < 1e-12:
            break
        current_f = current_f * (N / s)
        
        conv = np.convolve(current_f, current_f, mode='full')
        current_val = np.max(conv) / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_f = current_f.copy()
        
        # Get FFT
        F = np.fft.rfft(current_f, n=2*N)
        power = np.abs(F) ** 2
        
        # Identify frequency bins with high power that contribute to peak
        # The DC component is always present; focus on others
        if len(power) > 1:
            power_no_dc = power[1:]
            max_power = np.max(power_no_dc)
            if max_power > 1e-12:
                # Reduce high-power frequency components
                threshold = max_power * 0.9
                high_power_mask = power_no_dc >= threshold
                # Dampen these components in frequency domain
                F_damped = F.copy()
                F_damped[1:][high_power_mask] *= 0.95
                # Inverse FFT
                f_new = np.fft.irfft(F_damped, n=2*N)[:N]
                f_new = np.maximum(f_new, 0.0)
                s_new = np.sum(f_new)
                if s_new > 1e-12:
                    f_new = f_new * (N / s_new)
                    conv_new = np.convolve(f_new, f_new, mode='full')
                    val_new = np.max(conv_new) / (N * N)
                    if val_new < current_val:
                        current_f = f_new
                    else:
                        break
                else:
                    break
        else:
            break
    
    return best_f, best_val

def construct():
    start_time = time.time()
    rng = np.random.RandomState(123)
    
    # Try multiple initial guesses
    initial_guesses = [
        initial_guess_parabolic(),
        initial_guess_cosine(),
        np.ones(N),
        0.5 * np.ones(N) + 0.5 * initial_guess_parabolic(),
    ]
    
    best_f = None
    best_score = np.inf
    
    for guess in initial_guesses:
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        
        # Peak shaving
        refined_f, refined_score = peak_shaving(
            guess, start_time, TIME_LIMIT - 15, rng
        )
        
        if refined_score < best_score:
            best_score = refined_score
            best_f = refined_f
    
    # Spectral refinement on the best
    if best_f is not None and time.time() - start_time < TIME_LIMIT - 10:
        spec_f, spec_score = spectral_refinement(
            best_f, start_time, TIME_LIMIT - 5
        )
        if spec_score < best_score:
            best_score = spec_score
            best_f = spec_f
    
    # Final peak shaving with very low temperature for fine tuning
    if best_f is not None and time.time() - start_time < TIME_LIMIT - 5:
        fine_f, fine_score = peak_shaving(
            best_f, start_time, TIME_LIMIT - 1, rng
        )
        if fine_score < best_score:
            best_score = fine_score
            best_f = fine_f
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    
    # If all failed, return constant
    if best_f is None:
        best_f = np.ones(N)
    
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
