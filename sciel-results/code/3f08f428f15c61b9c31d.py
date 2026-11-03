"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. Use a smooth parametric family based on Gaussian bumps with multiple components.
2. Optimize using BFGS with analytic gradients for the smooth max surrogate.
3. Apply local refinement using sensitivity-based peak reduction with FFT-based convolution.
4. Add a final polish pass with adaptive step sizes.

Key improvements:
- Gradient-based optimization (BFGS) instead of derivative-free Nelder-Mead
- FFT-based autoconvolution for O(N log N) instead of O(N^2)
- Better parametric family with Gaussian bumps for smoother autoconvolution
- More aggressive local refinement with adaptive thresholds
"""

import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve
import time

N = 600
TIME_LIMIT = 135  # safety margin before 150s kill

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

def gaussian_basis(x, centers, widths, heights):
    """Generate a smooth nonnegative function from Gaussian bump components."""
    result = np.zeros_like(x)
    for c, w, h in zip(centers, widths, heights):
        result += h * np.exp(-0.5 * ((x - c) / (w / 2.0)) ** 2)
    return result

def parametric_to_step(params, n_components):
    """Convert parameters to a step function of length N."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    centers = params[:n_components]
    widths = params[n_components:2*n_components]
    heights = params[2*n_components:3*n_components]
    
    widths = np.abs(widths) + 0.01
    widths = np.minimum(widths, 0.5)
    heights = np.abs(heights)
    
    f = gaussian_basis(x, centers, widths, heights)
    f = np.maximum(f, 0.0)
    return f

def objective_and_grad(params, n_components, beta=10.0):
    """Objective and analytic gradient for the smooth max surrogate."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    centers = params[:n_components]
    widths = params[n_components:2*n_components]
    heights = params[2*n_components:3*n_components]
    
    w_eff = np.abs(widths) + 0.01
    w_eff = np.minimum(w_eff, 0.5)
    h_eff = np.abs(heights)
    
    # Compute f and derivatives of f w.r.t params
    f = np.zeros(N)
    df_dcenters = np.zeros((n_components, N))
    df_dwidths = np.zeros((n_components, N))
    df_dheights = np.zeros((n_components, N))
    
    for k in range(n_components):
        c = centers[k]
        w = w_eff[k]
        h = h_eff[k]
        
        z = (x - c) / w
        exp_term = np.exp(-0.5 * z**2)
        
        f += h * exp_term
        
        # df/dc
        df_dcenters[k] = -h * (x - c) / (w * w) * exp_term
        
        # df/dw
        df_dwidths[k] = h * (x - c)**2 / (w**3) * exp_term
        
        # df/dh
        df_dheights[k] = exp_term
    
    f = np.maximum(f, 0.0)
    
    conv = fftconvolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf, np.zeros_like(params)
    
    # Smooth max value
    x_max = np.max(conv)
    if x_max < 1e-15:
        val = 0.0
        p = np.zeros_like(conv)
    else:
        x_scaled = conv / x_max
        shifted = x_scaled - np.max(x_scaled)
        log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
        val = x_max * (1.0 + log_sum) / (s * s)
        
        # p_m = exp(beta * conv_m) / sum(exp(beta * conv))
        exp_conv = np.exp(beta * (conv - np.max(conv)))
        p = exp_conv / np.sum(exp_conv)
        
        # Compute dM/df
        # dM/df_j = sum_m p_m (f_{m-j} + f_{j-m})
        # A_j = sum_m p_m f_{m-j} (Correlation)
        # B_j = sum_m p_m f_{j-m} (Convolution)
        
        # Pad f and p to 2N for FFT
        size = 2 * N
        f_pad = np.zeros(size)
        f_pad[:N] = f
        p_pad = np.zeros(size)
        p_pad[:len(p)] = p
        
        F_fft = np.fft.rfft(f_pad, n=size)
        P_fft = np.fft.rfft(p_pad, n=size)
        
        # B = convolve(p, f)
        B_vals = np.fft.irfft(P_fft * F_fft, n=size)
        
        # A = correlate(p, f)
        # A_j = sum_m p_m f_{m-j}
        # This corresponds to IFFT(conj(P) * F)
        A_vals = np.fft.irfft(np.conj(P_fft) * F_fft, n=size)
        
        dM_df = np.zeros(N)
        dM_df += B_vals[0:N]
        dM_df += A_vals[0:N]
        
        # dval/df_j = (dM_df_j * s - 2 * M) / s^3
        # M = val * s^2
        # dval/df_j = dM_df_j / s^2 - 2 * val / s
        
        factor1 = 1.0 / (s * s)
        factor2 = 2.0 * val / s
        
        dval_df = dM_df * factor1 - factor2
        
        # Compute param gradients
        grad = np.zeros_like(params)
        
        for k in range(n_components):
            grad[k] = np.sum(dval_df * df_dcenters[k])
            grad[n_components + k] = np.sum(dval_df * df_dwidths[k])
            grad[2*n_components + k] = np.sum(dval_df * df_dheights[k])
            
    return val, grad

def local_refinement(h, start_time, max_time, alpha_init=0.1, threshold_frac=0.9):
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
        # For each bin j, sensitivity = sum over peak lags m of h[m-j]
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
            alpha = min(alpha * 1.04, 0.5)
        else:
            alpha *= 0.85
            if alpha < 1e-8:
                break
    
    return best_h, best_val

def polish_pass(h, start_time, max_time):
    """Final high-precision polish pass with very small step sizes."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.03
    
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
        
        threshold = max_c * 0.995
        peak_indices = np.where(conv >= threshold)[0]
        
        sensitivity = np.zeros(N)
        for m in peak_indices:
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            if j_min <= j_max:
                sensitivity[j_min:j_max+1] += current_h[m - j_max:m - j_min + 1]
        
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
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
            alpha = min(alpha * 1.02, 0.2)
        else:
            alpha *= 0.9
            if alpha < 1e-10:
                break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Try different numbers of Gaussian bump components
    for n_c in [2, 3, 4, 5]:
        if time.time() - start_time > TIME_LIMIT - 40:
            break
        
        rng = np.random.RandomState(42 + n_c)
        best_params = None
        best_obj = np.inf
        
        # Annealed beta: start lower, increase for sharper max approximation
        betas = [5.0, 10.0, 20.0]
        
        for beta in betas:
            if time.time() - start_time > TIME_LIMIT - 20:
                break
            
            for restart in range(12):
                if time.time() - start_time > TIME_LIMIT - 15:
                    break
                
                # Structured initialization
                if restart == 0:
                    # Symmetric initialization
                    centers = np.linspace(-0.15, 0.15, n_c)
                    widths = np.full(n_c, 0.12)
                    heights = np.full(n_c, 1.0)
                elif restart == 1:
                    # Asymmetric with varying widths
                    centers = rng.uniform(-0.2, 0.2, size=n_c)
                    widths = rng.uniform(0.04, 0.2, size=n_c)
                    heights = rng.uniform(0.5, 2.0, size=n_c)
                elif restart == 2:
                    # Concentrated at center
                    centers = np.zeros(n_c) + rng.uniform(-0.05, 0.05, size=n_c)
                    widths = rng.uniform(0.03, 0.1, size=n_c)
                    heights = rng.uniform(0.8, 2.5, size=n_c)
                else:
                    centers = rng.uniform(-0.2, 0.2, size=n_c)
                    widths = rng.uniform(0.02, 0.25, size=n_c)
                    heights = rng.uniform(0.3, 3.0, size=n_c)
                
                x0 = np.concatenate([centers, widths, heights])
                
                try:
                    result = minimize(
                        objective_and_grad, x0, args=(n_c, beta),
                        method='BFGS',
                        jac=True,
                        options={'maxiter': 5000, 'gtol': 1e-6}
                    )
                    if result.fun < best_obj:
                        best_obj = result.fun
                        best_params = result.x
                except Exception:
                    pass
        
        if best_params is not None:
            f = parametric_to_step(best_params, n_c)
            actual_score = compute_score(f)
            if actual_score < best_score:
                best_f = f
                best_score = actual_score
    
    # Local refinement: sensitivity-based peak reduction
    if time.time() - start_time < TIME_LIMIT - 25:
        refined_f, refined_score = local_refinement(
            best_f, start_time, TIME_LIMIT - 10
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Final polish pass
    if time.time() - start_time < TIME_LIMIT - 5:
        polished_f, polished_score = polish_pass(
            best_f, start_time, TIME_LIMIT - 2
        )
        if polished_score < best_score:
            best_f = polished_f
            best_score = polished_score
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
