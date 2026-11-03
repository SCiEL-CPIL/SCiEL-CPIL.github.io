"""
Optimize a nonnegative step function on [-1/4, 1/4] to minimize the peak of its autoconvolution.

Strategy:
1. Represent the step function as the squared magnitude of a complex-valued analytic signal
   defined by a small number of Gaussian modes with fixed centers.
2. Optimize the real and imaginary coefficients of these modes via Nelder-Mead to minimize
   the peak of the autoconvolution, leveraging the inherent non-negativity and smoothness
   of the squared modulus.
3. Apply local refinement using sensitivity-based peak reduction (from inspiration 1).
4. Add a final polish pass with tighter tolerances.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 600
H = 1.0 / (2 * N)

def _fft_autoconvolve(x):
    """Compute the autoconvolution of x using FFT."""
    n = len(x)
    f = np.fft.rfft(x, n=2 * n)
    conv = np.fft.irfft(f * f, n=2 * n)
    return conv

def _score(x):
    """Compute the upper bound on C1 from the step function x."""
    conv = _fft_autoconvolve(x)
    peak = np.max(conv) * H
    return peak

def _smooth_max(values, sigma=0.01):
    """Log-sum-exp based smooth maximum."""
    m = np.max(values)
    if m < 1e-15:
        return 0.0
    shifted = (values - m) / sigma
    return m + sigma * np.log(np.sum(np.exp(shifted)))

def _parametric_to_step(params, n_modes, t):
    """Convert parameters to a step function of length N.
    
    params: [re_0, im_0, re_1, im_1, ...] for each mode
    The step function is |sum_k (re_k + i*im_k) * exp(-0.5*((t-c_k)/w_k)^2 * (1+1j*phi_k))|^2
    """
    re_coeffs = params[:n_modes]
    im_coeffs = params[n_modes:2*n_modes]
    
    # Fixed centers and widths for Gaussian modes
    centers = np.linspace(-0.2, 0.2, n_modes)
    widths = np.linspace(0.05, 0.15, n_modes)
    phases = np.linspace(0, np.pi, n_modes)
    
    # Compute complex signal
    signal = np.zeros(N, dtype=np.complex128)
    for k in range(n_modes):
        coeff = re_coeffs[k] + 1j * im_coeffs[k]
        gaussian = np.exp(-0.5 * ((t - centers[k]) / widths[k]) ** 2 * (1 + 1j * phases[k]))
        signal += coeff * gaussian
    
    # Squared magnitude gives nonnegative step function
    x = np.abs(signal) ** 2
    x = np.maximum(x, 0)
    return x

def _objective(params, n_modes, t, target_sum):
    """Objective function for optimization."""
    x = _parametric_to_step(params, n_modes, t)
    sum_x = np.sum(x)
    if sum_x < 1e-12:
        return 1e9
    x = x * (target_sum / sum_x)
    conv = _fft_autoconvolve(x)
    peak = np.max(conv) * H
    return peak

def _smooth_objective(params, n_modes, t, target_sum, sigma=0.005):
    """Smooth objective function for optimization."""
    x = _parametric_to_step(params, n_modes, t)
    sum_x = np.sum(x)
    if sum_x < 1e-12:
        return 1e9
    x = x * (target_sum / sum_x)
    conv = _fft_autoconvolve(x)
    conv_scaled = conv * H
    return _smooth_max(conv_scaled, sigma)

def _local_refinement(h, start_time, max_time, alpha_init=0.08, threshold_frac=0.92):
    """Local refinement using sensitivity-based peak reduction."""
    best_h = h.copy()
    best_val = _score(best_h)
    current_h = h.copy()
    alpha = alpha_init
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = _fft_autoconvolve(current_h)
        conv_scaled = conv * H
        max_c = np.max(conv_scaled)
        current_val = max_c
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        # Find lags where conv is close to max
        threshold = max_c * threshold_frac
        peak_indices = np.where(conv_scaled >= threshold)[0]
        
        # Compute sensitivity: for each bin j, how much does it contribute to peak lags?
        sensitivity = np.zeros(N)
        for m in peak_indices:
            # m is index in conv (length 2N-1), corresponds to lag m - (N-1)
            # For autoconvolution, conv[m] = sum_i x[i]*x[m-i] (with appropriate bounds)
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            for j in range(j_min, j_max + 1):
                sensitivity[j] += current_h[m - j]
        
        # Normalize sensitivity
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        # Reduce bins with high sensitivity (they contribute to peaks)
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = _fft_autoconvolve(h_new)
        val_new = np.max(conv_new) * H
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.03, 0.4)
        else:
            alpha *= 0.88
            if alpha < 1e-8:
                break
    
    return best_h, best_val

def _polish_pass(h, start_time, max_time):
    """Final high-precision polish pass with very small step sizes."""
    best_h = h.copy()
    best_val = _score(best_h)
    current_h = h.copy()
    alpha = 0.02
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = _fft_autoconvolve(current_h)
        conv_scaled = conv * H
        max_c = np.max(conv_scaled)
        current_val = max_c
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        # Very tight threshold for polish
        threshold = max_c * 0.99
        peak_indices = np.where(conv_scaled >= threshold)[0]
        
        sensitivity = np.zeros(N)
        for m in peak_indices:
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            for j in range(j_min, j_max + 1):
                sensitivity[j] += current_h[m - j]
        
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = _fft_autoconvolve(h_new)
        val_new = np.max(conv_new) * H
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.02, 0.15)
        else:
            alpha *= 0.9
            if alpha < 1e-10:
                break
    
    return best_h, best_val

def construct():
    """Construct a nonnegative step function with 600 intervals on [-1/4, 1/4]
    to minimize the normalized maximum of its autoconvolution.
    
    Returns: list of 600 nonnegative step heights.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    t = np.linspace(-1/4, 1/4, N, endpoint=False)
    target_sum = N / 2.0  # Target sum for normalization
    
    # Initial best: constant function
    best_x = np.ones(N) * (target_sum / N)
    best_score = _score(best_x)
    
    # Try different numbers of Gaussian modes with Nelder-Mead
    n_modes_list = [3, 4, 5]
    
    for n_modes in n_modes_list:
        if time.time() > deadline - 40.0:
            break
        
        # Initial parameters: small random values
        rng = np.random.RandomState(42 + n_modes)
        best_params = None
        best_obj = np.inf
        
        for restart in range(5):
            if time.time() > deadline - 20.0:
                break
            
            if restart == 0:
                # Symmetric initialization
                re_init = np.ones(n_modes) * 0.5
                im_init = np.zeros(n_modes)
            else:
                re_init = rng.uniform(-1, 1, size=n_modes)
                im_init = rng.uniform(-1, 1, size=n_modes)
            
            params0 = np.concatenate([re_init, im_init])
            
            try:
                result = minimize(
                    _objective, params0,
                    args=(n_modes, t, target_sum),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                obj_val = result.fun
                if obj_val < best_obj:
                    best_obj = obj_val
                    best_params = result.x.copy()
            except Exception:
                pass
        
        if best_params is not None:
            x_candidate = _parametric_to_step(best_params, n_modes, t)
            sum_x = np.sum(x_candidate)
            if sum_x > 1e-12:
                x_candidate = x_candidate * (target_sum / sum_x)
                x_candidate = np.maximum(x_candidate, 0)
                score_candidate = _score(x_candidate)
                if score_candidate < best_score:
                    best_score = score_candidate
                    best_x = x_candidate.copy()
    
    # Local refinement using sensitivity-based peak reduction
    try:
        if time.time() < deadline - 20.0:
            best_x, best_score = _local_refinement(best_x, start_time, deadline - 15.0)
    except Exception:
        pass
    
    # Final polish pass
    try:
        if time.time() < deadline - 5.0:
            best_x, best_score = _polish_pass(best_x, start_time, deadline - 2.0)
    except Exception:
        pass
    
    # Ensure nonnegative
    best_x = np.maximum(best_x, 0)
    
    return [float(v) for v in best_x]
