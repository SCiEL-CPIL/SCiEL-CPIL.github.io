"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Spectral coordinate descent in the DFT domain.
- Parameterize the step function by its DFT coefficients (amplitude + phase).
- Enforce non-negativity via soft-plus projection in time domain.
- Directly minimize the maximum of the autoconvolution by adjusting spectral
  coefficients that contribute most to peak lags.
- Use FFT for O(N log N) autoconvolution and sensitivity computation.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 135  # safety margin before 150s kill

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

def softplus(x, beta=10.0):
    """Numerically stable softplus for non-negativity projection."""
    return np.where(x > 1.0 / beta, x, np.log1p(np.exp(beta * x)) / beta)

def spectral_to_time(amps, phases, n_fft):
    """Reconstruct time-domain signal from spectral amplitude and phase."""
    H = amps * np.exp(1j * phases)
    H_full = np.zeros(n_fft // 2 + 1, dtype=np.complex128)
    H_full[:len(amps)] = H
    # Enforce Hermitian symmetry
    H_full[1:-1] = np.conj(H_full[1:-1][::-1])
    f = np.fft.irfft(H_full, n=n_fft)[:N]
    return f

def autoconv_from_fft(f, n_fft):
    """Compute autoconvolution via FFT."""
    f_hat = np.fft.rfft(f, n=n_fft)
    conv = np.fft.irfft(f_hat * f_hat, n=n_fft)[:2 * N - 1]
    return conv

def spectral_coordinate_descent(f_init, start_time, max_time):
    """
    Coordinate descent in the DFT domain to minimize the autoconvolution peak.
    
    For each frequency bin, adjust its amplitude and phase to reduce the
    contribution to the dominant autoconvolution peaks while maintaining
    non-negativity of the time-domain signal.
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    
    # Initialize spectrum from time-domain signal
    f = f_init.copy()
    s = np.sum(f)
    if s < 1e-12:
        return f, compute_score(f)
    f = f * (N / s)
    
    best_f = f.copy()
    best_val = compute_score(f)
    
    f_hat = np.fft.rfft(f, n=n_fft)
    amps = np.abs(f_hat)
    phases = np.angle(f_hat)
    
    # Precompute which lags are "peaks" and their weights
    conv = autoconv_from_fft(f, n_fft)
    
    lr_amps = 0.05
    lr_phases = 0.02
    peak_threshold = 0.85
    smooth_beta = 8.0
    
    while time.time() - start_time < max_time:
        # Update peak mask based on current autoconvolution
        conv_max = np.max(conv)
        if conv_max < 1e-15:
            break
        threshold = conv_max * peak_threshold
        peak_mask = (conv >= threshold).astype(np.float64)
        
        # Softmax weights over peaks for smooth sensitivity
        shifted = conv - conv_max
        exp_w = np.exp(smooth_beta * shifted)
        w = exp_w / np.sum(exp_w)
        
        # Sensitivity: for each time bin j, how much does it contribute to peaks?
        # d(conv[m])/d(f[j]) = f[m-j]
        # sens[j] = sum_m w[m] * f[m-j] = cross-correlation of w with f
        w_hat = np.fft.rfft(peak_mask, n=n_fft)
        f_hat_current = np.fft.rfft(f, n=n_fft)
        sens = np.fft.irfft(w_hat * np.conj(f_hat_current), n=n_fft)[:N]
        
        # Normalize sensitivity
        sens_mean = np.mean(sens)
        sens_std = np.std(sens) + 1e-12
        sens_norm = (sens - sens_mean) / sens_std
        
        # Spectral coordinate descent: for each frequency bin, compute
        # the gradient of the objective w.r.t. that bin's amplitude and phase
        # and take a small step.
        
        # Time-domain gradient of smoothed max:
        # grad_f[j] = 2 * sens[j] / s^2 - 2*K/(s^3)
        K = conv_max + (1.0 / smooth_beta) * np.log(np.sum(exp_w))
        s2 = np.sum(f) ** 2
        grad_f = 2.0 * sens / s2 - (2.0 * K / (s2 * np.sum(f))) * np.ones(N)
        
        # Project gradient to frequency domain
        grad_f_hat = np.fft.rfft(grad_f, n=n_fft)
        
        # Update amplitudes: reduce amplitudes where gradient is large
        # and increase where gradient is small (relative to mean)
        grad_amps = np.real(f_hat * np.conj(grad_f_hat)) / (amps + 1e-12)
        grad_phases = np.imag(f_hat * np.conj(grad_f_hat)) / (amps + 1e-12)
        
        # Normalize gradients for stable updates
        ga_mean = np.mean(grad_amps)
        ga_std = np.std(grad_amps) + 1e-12
        ga_norm = (grad_amps - ga_mean) / ga_std
        
        gp_mean = np.mean(grad_phases)
        gp_std = np.std(grad_phases) + 1e-12
        gp_norm = (grad_phases - gp_mean) / gp_std
        
        # Step in spectral domain
        amps_new = amps * (1.0 - lr_amps * ga_norm)
        amps_new = np.maximum(amps_new, 0.0)
        phases_new = phases - lr_phases * gp_norm
        
        # Reconstruct and project to non-negativity
        f_new = spectral_to_time(amps_new, phases_new, n_fft)
        f_new = softplus(f_new, beta=5.0)
        
        # Renormalize
        s_new = np.sum(f_new)
        if s_new < 1e-12:
            break
        f_new = f_new * (N / s_new)
        
        # Evaluate
        conv_new = autoconv_from_fft(f_new, n_fft)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < best_val:
            best_val = val_new
            best_f = f_new.copy()
            f = f_new
            f_hat = np.fft.rfft(f, n=n_fft)
            amps = np.abs(f_hat)
            phases = np.angle(f_hat)
            conv = conv_new
            # Gradually increase learning rates
            lr_amps = min(lr_amps * 1.02, 0.15)
            lr_phases = min(lr_phases * 1.02, 0.08)
        else:
            # Rejected: decay learning rates
            lr_amps *= 0.9
            lr_phases *= 0.9
            if lr_amps < 1e-6:
                break
    
    return best_f, best_val

def time_domain_refinement(f, start_time, max_time):
    """
    Direct time-domain refinement using sensitivity-based peak reduction.
    Complements the spectral approach by operating directly on time bins.
    """
    best_f = f.copy()
    best_val = compute_score(f)
    current_f = f.copy()
    
    n_fft = 1 << (2 * N - 1).bit_length()
    alpha = 0.03
    threshold_frac = 0.90
    
    while time.time() - start_time < max_time:
        s = np.sum(current_f)
        if s < 1e-12:
            break
        current_f = current_f * (N / s)
        
        conv = autoconv_from_fft(current_f, n_fft)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_f = current_f.copy()
        
        threshold = max_c * threshold_frac
        peak_mask = (conv >= threshold).astype(np.float64)
        
        # Sensitivity via FFT cross-correlation
        pm_hat = np.fft.rfft(peak_mask, n=n_fft)
        f_hat = np.fft.rfft(current_f, n=n_fft)
        sens = np.fft.irfft(pm_hat * np.conj(f_hat), n=n_fft)[:N]
        
        sens_mean = np.mean(sens)
        sens_std = np.std(sens) + 1e-12
        sens_norm = (sens - sens_mean) / sens_std
        
        # Reduce high-sensitivity bins
        f_new = current_f * (1.0 - alpha * sens_norm)
        f_new = np.maximum(f_new, 0.0)
        
        s_new = np.sum(f_new)
        if s_new < 1e-12:
            break
        f_new = f_new * (N / s_new)
        
        conv_new = autoconv_from_fft(f_new, n_fft)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_f = f_new
            alpha = min(alpha * 1.03, 0.25)
        else:
            alpha *= 0.85
            if alpha < 1e-8:
                break
    
    return best_f, best_val

def generate_initial_candidates(rng, x):
    """Generate diverse initial candidates using different structural families."""
    candidates = []
    
    # Gaussian
    f0 = np.exp(-15 * x**2)
    f0 = f0 / np.sum(f0)
    candidates.append(f0)
    
    # Cosine bump: single wide bump
    f1 = np.maximum(np.cos(np.pi * x / 0.2), 0.0) ** 2
    s1 = np.sum(f1)
    if s1 > 1e-12:
        candidates.append(f1 / s1)
    
    # Two symmetric bumps
    f2 = np.zeros_like(x)
    for c in [-0.08, 0.08]:
        dist = np.abs(x - c)
        mask = dist <= 0.06
        f2[mask] += np.cos(np.pi * (x[mask] - c) / 0.12) ** 2
    s2 = np.sum(f2)
    if s2 > 1e-12:
        candidates.append(f2 / s2)
    
    # Three bumps
    f3 = np.zeros_like(x)
    for c in [-0.12, 0.0, 0.12]:
        dist = np.abs(x - c)
        mask = dist <= 0.05
        f3[mask] += np.cos(np.pi * (x[mask] - c) / 0.10) ** 2
    s3 = np.sum(f3)
    if s3 > 1e-12:
        candidates.append(f3 / s3)
    
    # Random smooth function via low-frequency DFT
    n_fft = 1 << (2 * N - 1).bit_length()
    for trial in range(5):
        f_rand = np.zeros(N)
        # Use only low-frequency components
        for k in range(1, 30):
            amp = rng.uniform(0.1, 1.0)
            phase = rng.uniform(0, 2 * np.pi)
            f_rand += amp * np.cos(2 * np.pi * k * np.arange(N) / N + phase)
        f_rand = np.maximum(f_rand, 0.0)
        s_rand = np.sum(f_rand)
        if s_rand > 1e-12:
            candidates.append(f_rand / s_rand)
    
    return candidates

def construct():
    start_time = time.time()
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    
    best_f = np.ones(N) / N
    best_score = compute_score(best_f)
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Generate diverse initial candidates
    candidates = generate_initial_candidates(rng, x)
    
    # Phase 2: Spectral coordinate descent on each candidate
    for idx, f_init in enumerate(candidates):
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        
        try:
            f_res, score_res = spectral_coordinate_descent(
                f_init, start_time, TIME_LIMIT - 30
            )
            if score_res < best_score:
                best_score = score_res
                best_f = f_res.copy()
        except Exception:
            pass
    
    # Phase 3: Time-domain refinement on the best candidate
    if time.time() - start_time < TIME_LIMIT - 10:
        refined_f, refined_score = time_domain_refinement(
            best_f, start_time, TIME_LIMIT - 5
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Phase 4: Final spectral polish
    if time.time() - start_time < TIME_LIMIT - 5:
        polished_f, polished_score = spectral_coordinate_descent(
            best_f, start_time, TIME_LIMIT - 3
        )
        if polished_score < best_score:
            best_f = polished_f
            best_score = polished_score
    
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
