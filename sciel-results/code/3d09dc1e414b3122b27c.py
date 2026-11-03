"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Structural change: Use a frequency-domain (spectral) parameterization with
coordinate descent. Instead of optimizing in the time domain with subgradient
methods, we optimize the DFT coefficients directly. This allows us to directly
control the shape of the autoconvolution spectrum, and we use a smooth surrogate
for the peak minimization.

Key ideas:
1. Represent f via its DFT coefficients F[k] (Hermitian symmetry).
2. The autoconvolution spectrum is |F[k]|^2.
3. Minimize the peak of the inverse DFT of |F|^2 subject to f >= 0.
4. Use coordinate descent on the magnitude and phase of F[k].
5. Use a smooth max surrogate for the peak.
6. Apply projection onto f >= 0 after each update.
"""

import numpy as np
from scipy.signal import fftconvolve
import time

N = 600
TIME_LIMIT = 145  # safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = fftconvolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def smooth_max(x, beta):
    """Smooth approximation to max using log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0
    x_scaled = x / x_max
    shifted = x_scaled - np.max(x_scaled)
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return x_max * (1.0 + log_sum)

def spectral_coordinate_descent(f_init, start_time, max_time, beta=20.0):
    """
    Coordinate descent in the frequency domain to minimize the peak of autoconvolution.
    
    We work with the DFT coefficients F[k] for k = 0, ..., M-1 where M is the FFT size.
    For real f, F has Hermitian symmetry: F[M-k] = conj(F[k]).
    
    The autoconvolution c = f * f has DFT C[k] = |F[k]|^2.
    We want to minimize max(c) = max(ifft(|F|^2)).
    
    Strategy:
    - Initialize F from f_init.
    - For each frequency bin k, adjust |F[k]| and phase[F[k]] to reduce the peak.
    - After each update, project back to f >= 0.
    """
    M = 2048  # FFT size for zero padding
    f = f_init.copy()
    s = np.sum(f)
    if s < 1e-12:
        return f, np.inf
    f = f * (N / s)  # normalize
    
    F = np.fft.fft(f, M)
    best_f = f.copy()
    best_val = compute_score(f)
    best_F = F.copy()
    
    lr = 0.05
    n_updates = 0
    
    while time.time() - start_time < max_time:
        # Compute current autoconvolution
        C_spec = np.abs(F)**2
        c = np.fft.ifft(C_spec).real
        max_c = np.max(c)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_f = np.fft.ifft(F, M).real[:N]
            best_f = np.maximum(best_f, 0.0)
            best_F = F.copy()
        
        # Find peak location
        k_peak = np.argmax(c)
        
        # For each frequency bin, try to adjust to reduce the peak
        # We use a smooth surrogate gradient approach
        # grad of smooth_max(c) w.r.t. c is approximately exp(beta*(c - max_c)) / sum(exp(beta*(c - max_c)))
        c_shifted = c - max_c
        weights = np.exp(beta * c_shifted)
        weights /= np.sum(weights)
        
        # Gradient w.r.t. F[k]
        # c = ifft(|F|^2)
        # d(c[m])/d(F[k]) = ?
        # This is complex. Let's use a simpler approach:
        # For each k, compute the contribution of F[k] to the peak.
        
        # Actually, let's use a different approach:
        # We know that c = ifft(|F|^2).
        # To reduce the peak at k_peak, we want to reduce |F[k]|^2 at frequencies
        # that contribute to the peak.
        
        # The contribution of F[k] to c[m] is:
        # c[m] = (1/M) * sum_k |F[k]|^2 * exp(2*pi*i*k*m/M)
        # So d(c[m])/d(|F[k]|^2) = (1/M) * exp(2*pi*i*k*m/M)
        
        # To reduce c[k_peak], we should reduce |F[k]|^2 for k where
        # Re(exp(2*pi*i*k*k_peak/M)) is large positive.
        
        # Let's compute the "importance" of each frequency bin for the peak
        n = np.arange(M)
        peak_phase = np.exp(2j * np.pi * k_peak * n / M)
        importance = np.real(peak_phase)
        
        # Adjust |F| based on importance
        # Reduce |F| where importance is high, increase where low
        # But we need to maintain f >= 0
        
        F_new = F.copy()
        for k in range(M // 2 + 1):
            if k == 0 or k == M // 2:
                # DC and Nyquist: real only
                F_new[k] = F[k] - lr * importance[k] * np.abs(F[k])
                if F_new[k].real < 0:
                    F_new[k] = 0
            else:
                # Complex: adjust magnitude
                mag = np.abs(F[k])
                phase = np.angle(F[k])
                new_mag = mag * (1 - lr * importance[k])
                new_mag = max(new_mag, 0)
                F_new[k] = new_mag * np.exp(1j * phase)
                F_new[M-k] = np.conj(F_new[k])
        
        # Project to f >= 0
        f_new = np.fft.ifft(F_new, M).real[:N]
        f_new = np.maximum(f_new, 0.0)
        s_new = np.sum(f_new)
        if s_new < 1e-12:
            break
        f_new = f_new * (N / s_new)
        
        F_new = np.fft.fft(f_new, M)
        
        # Check improvement
        C_new = np.abs(F_new)**2
        c_new = np.fft.ifft(C_new).real
        val_new = np.max(c_new) / (N * N)
        
        if val_new < current_val:
            F = F_new
            lr = min(lr * 1.03, 0.3)
        else:
            lr *= 0.9
            if lr < 1e-8:
                break
            # Try a different strategy: random perturbation
            if np.random.random() < 0.1:
                F = F * (1 + 0.01 * np.random.randn(M))
                f_temp = np.fft.ifft(F, M).real[:N]
                f_temp = np.maximum(f_temp, 0.0)
                s_temp = np.sum(f_temp)
                if s_temp > 1e-12:
                    f_temp = f_temp * (N / s_temp)
                    F = np.fft.fft(f_temp, M)
        
        n_updates += 1
        if n_updates % 100 == 0:
            # Periodic projection to ensure f >= 0
            f_temp = np.fft.ifft(F, M).real[:N]
            f_temp = np.maximum(f_temp, 0.0)
            s_temp = np.sum(f_temp)
            if s_temp > 1e-12:
                f_temp = f_temp * (N / s_temp)
                F = np.fft.fft(f_temp, M)
    
    return best_f, best_val

def local_refinement(h, start_time, max_time, alpha_init=0.05, threshold_frac=0.95):
    """Local refinement using sensitivity-based peak reduction with FFT."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = alpha_init
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = fftconvolve(current_h, current_h, mode='full')
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        # Find lags where conv is close to max
        threshold = max_c * threshold_frac
        peak_indices = np.where(conv >= threshold)[0]
        
        # Compute sensitivity using FFT-based approach
        sensitivity = np.zeros(N)
        for m in peak_indices:
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            if j_min <= j_max:
                sensitivity[j_min:j_max+1] += current_h[m - j_max:m - j_min + 1]
        
        # Normalize sensitivity
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        # Reduce bins with high sensitivity
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fftconvolve(h_new, h_new, mode='full')
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.04, 0.3)
        else:
            alpha *= 0.85
            if alpha < 1e-8:
                break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Spectral coordinate descent
    if time.time() - start_time < TIME_LIMIT - 15:
        refined_f, refined_score = spectral_coordinate_descent(
            best_f, start_time, TIME_LIMIT - 10
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Local refinement
    if time.time() - start_time < TIME_LIMIT - 10:
        refined_f, refined_score = local_refinement(
            best_f, start_time, TIME_LIMIT - 5
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Try a few random initializations
    rng = np.random.RandomState(42)
    for i in range(3):
        if time.time() - start_time > TIME_LIMIT - 15:
            break
        # Random smooth initialization
        f_init = np.ones(N) * rng.uniform(0.5, 1.5)
        noise = rng.normal(0, 0.1, N)
        kernel = np.ones(10) / 10
        noise = np.convolve(noise, kernel, mode='same')
        f_init = f_init + noise
        f_init = np.maximum(f_init, 0)
        
        refined_f, refined_score = spectral_coordinate_descent(
            f_init, start_time, TIME_LIMIT - 10
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
        
        if time.time() - start_time < TIME_LIMIT - 5:
            refined_f, refined_score = local_refinement(
                best_f, start_time, TIME_LIMIT - 5
            )
            if refined_score < best_score:
                best_f = refined_f
                best_score = refined_score
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
