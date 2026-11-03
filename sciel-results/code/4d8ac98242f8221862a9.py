"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. Use a smooth parametric family based on Gaussian bumps with multiple components.
2. Optimize using BFGS with analytic gradients for the smooth max surrogate.
3. Apply local refinement using sensitivity-based peak reduction with FFT-based convolution.
4. Add a final polish pass with adaptive step sizes.
5. Incorporate spectral initialization and peak shaving with simulated annealing from inspiration.

Key improvements:
- Gradient-based optimization (BFGS) instead of derivative-free Nelder-Mead
- FFT-based autoconvolution for O(N log N) instead of O(N^2)
- Better parametric family with Gaussian bumps for smoother autoconvolution
- More aggressive local refinement with adaptive thresholds
- Spectral initialization for better starting points
- Simulated annealing peak shaving for escaping local minima
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
    
    # Compute f and gradients w.r.t parameters
    f = np.zeros(N)
    # Store gradients for each param
    # Params: c1..cn, w1..wn, h1..hn
    grad_f = np.zeros((3*n_components, N))
    
    for k in range(n_components):
        c = centers[k]
        w = w_eff[k]
        h = h_eff[k]
        
        # g = exp(-0.5 * ((x-c)/w/2)^2)
        # Let u = (x-c) / (w/2) = 2(x-c)/w
        u = 2.0 * (x - c) / w
        g = np.exp(-0.5 * u * u)
        
        f += h * g
        
        # Gradients
        # d f / d h_k = g
        grad_f[2*n_components + k] += g
        
        # d f / d c_k = h * g * (2(x-c)/w) = h * g * u
        grad_f[k] += h * g * u
        
        # d f / d w_k = h * g * (4(x-c)^2/w^3)
        # Note: u = 2(x-c)/w => (x-c) = u*w/2 => (x-c)^2 = u^2 w^2 / 4
        # Term = h * g * 4 * (u^2 w^2 / 4) / w^3 = h * g * u^2 / w
        grad_f[n_components + k] += h * g * (u * u) / w
    
    f = np.maximum(f, 0.0)
    
    # If f was clipped, gradients are technically 0 where f<0, but since f is sum of pos gaussians,
    # and we initialized with pos heights, f should be > 0. 
    # If any f < 0, we should handle, but max(0, f) ensures non-negativity.
    # For gradient, if f_i > 0, grad is valid. If f_i == 0, grad should be 0.
    # Since f is sum of positives, f_i=0 only if all components are 0 there.
    # We'll assume f > 0 for simplicity, or mask if necessary.
    
    s = np.sum(f)
    if s < 1e-15:
        return np.inf, np.zeros_like(params)
    
    # Compute conv and grad_conv
    # conv = f * f
    # grad_conv_theta = 2 * (f * grad_f_theta)
    
    fft_f = np.fft.fft(f, n=2*N)
    
    conv = np.fft.ifft(fft_f * fft_f).real[:2*N-1]
    
    # Smooth max of conv
    # val = smooth_max(conv, beta) / s^2
    # grad_val = (1/s^2) * grad_smooth_max - (2/s^3) * smooth_max * grad_s
    
    # grad_s = sum(grad_f_theta)
    grad_s = np.sum(grad_f, axis=1)
    
    # grad_smooth_max
    # smooth_max(x) = 1/beta * log(sum(exp(beta * x)))
    # grad = sum(p_i * grad_x_i) where p_i = exp(beta*x_i)/sum(exp(beta*x_j))
    
    x_max = np.max(conv)
    if x_max < 1e-15:
        # If conv is very small, gradient is ill-defined or 0
        return smooth_max(conv, beta) / (s * s), np.zeros_like(params)
        
    shifted = conv - x_max
    exp_shifted = np.exp(beta * shifted)
    sum_exp = np.sum(exp_shifted)
    p = exp_shifted / sum_exp
    
    smooth_max_val = x_max + np.log(sum_exp) / beta
    val = smooth_max_val / (s * s)
    
    # We need sum(p_i * grad_conv_i) for each param
    # grad_conv_theta = 2 * IFFT(fft_f * fft_grad_f_theta)
    # We can compute this in frequency domain if we have fft_grad_f_theta
    
    grad_val = np.zeros_like(params)
    
    for i in range(3 * n_components):
        # Compute fft of grad_f[i]
        fft_gf = np.fft.fft(grad_f[i], n=2*N)
        # grad_conv_i = 2 * IFFT(fft_f * fft_gf)
        grad_conv_i = np.fft.ifft(fft_f * fft_gf).real[:2*N-1] * 2.0
        
        # grad_smooth_max_i = sum(p * grad_conv_i)
        dot_product = np.dot(p, grad_conv_i)
        
        # Total gradient
        grad_val[i] = (dot_product / (s * s)) - (2.0 * smooth_max_val * grad_s[i] / (s * s * s))
        
    return val, grad_val

def local_refinement(h, start_time, max_time, alpha_init=0.1, threshold_frac=0.9):
    """Local refinement using FFT-based sensitivity and adaptive step sizes."""
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
        peak_mask = np.zeros(len(conv), dtype=bool)
        peak_mask[conv >= threshold] = True
        
        # Compute sensitivity using FFT: sensitivity[j] = sum_m peak_mask[m] * h[m-j]
        # This is a cross-correlation of h with the peak indicator
        # Use FFT for O(N log N) instead of O(N * |peaks|)
        fft_h = np.fft.fft(current_h, n=2*N)
        fft_peak = np.fft.fft(peak_mask[:2*N-1], n=2*N)
        # Cross-correlation: IFFT(fft_h * conj(fft_peak))
        sens_full = np.fft.ifft(fft_h * np.conj(fft_peak)).real
        sensitivity = sens_full[:N]
        
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
            alpha = min(alpha * 1.05, 0.6)
        else:
            alpha *= 0.82
            if alpha < 1e-8:
                break
    
    return best_h, best_val

def polish_pass(h, start_time, max_time):
    """Final high-precision polish pass with FFT-based sensitivity and fine steps."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.02
    
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
        
        # Use tighter threshold for polish
        threshold = max_c * 0.999
        peak_mask = np.zeros(len(conv), dtype=bool)
        peak_mask[conv >= threshold] = True
        
        # FFT-based sensitivity computation
        fft_h = np.fft.fft(current_h, n=2*N)
        fft_peak = np.fft.fft(peak_mask[:2*N-1], n=2*N)
        sens_full = np.fft.ifft(fft_h * np.conj(fft_peak)).real
        sensitivity = sens_full[:N]
        
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
            alpha = min(alpha * 1.015, 0.15)
        else:
            alpha *= 0.88
            if alpha < 1e-10:
                break
    
    return best_h, best_val

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

def spectral_refinement(f, start_time, max_time):
    """Refine using frequency domain insights.
    
    The autoconvolution peak is related to the L-infinity norm of the inverse
    Fourier transform of |F|^2. We can try to flatten the power spectrum.
    """
    best_f = f.copy()
    best_val = compute_score(f)
    current_f = f.copy()
    
    for iteration in range(2000):
        if time.time() - start_time > max_time:
            break
        
        s = np.sum(current_f)
        if s < 1e-12:
            break
        current_f = current_f * (N / s)
        
        conv = fftconvolve(current_f, current_f, mode='full')
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
                    conv_new = fftconvolve(f_new, f_new, mode='full')
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
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Try different initial guesses from inspiration
    initial_guesses = [
        initial_guess_parabolic(),
        initial_guess_cosine(),
        np.ones(N),
        0.5 * np.ones(N) + 0.5 * initial_guess_parabolic(),
    ]
    
    # Try different numbers of Gaussian bump components
    for n_c in [2, 3, 4, 5]:
        if time.time() - start_time > TIME_LIMIT - 40:
            break
        
        rng = np.random.RandomState(42 + n_c)
        best_params = None
        best_obj = np.inf
        
        # Annealed beta: start lower, increase for sharper max approximation
        betas = [3.0, 8.0, 15.0, 30.0, 60.0]
        
        for beta in betas:
            if time.time() - start_time > TIME_LIMIT - 25:
                break
            
            # Increase restarts since analytic gradient is faster
            for restart in range(15):
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
                elif restart == 3:
                    # Wide, low bumps
                    centers = rng.uniform(-0.22, 0.22, size=n_c)
                    widths = rng.uniform(0.15, 0.3, size=n_c)
                    heights = rng.uniform(0.3, 1.0, size=n_c)
                elif restart == 4:
                    # Narrow, tall bumps
                    centers = rng.uniform(-0.1, 0.1, size=n_c)
                    widths = rng.uniform(0.02, 0.06, size=n_c)
                    heights = rng.uniform(1.5, 4.0, size=n_c)
                else:
                    centers = rng.uniform(-0.22, 0.22, size=n_c)
                    widths = rng.uniform(0.02, 0.28, size=n_c)
                    heights = rng.uniform(0.2, 3.5, size=n_c)
                
                x0 = np.concatenate([centers, widths, heights])
                
                try:
                    result = minimize(
                        objective_and_grad, x0, args=(n_c, beta),
                        method='BFGS',
                        jac=True,
                        options={'maxiter': 8000, 'gtol': 1e-9, 'factr': 1e-13}
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
    
    # Try initial guesses with spectral refinement
    for guess in initial_guesses:
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        
        refined_f, refined_score = spectral_refinement(
            guess, start_time, TIME_LIMIT - 20
        )
        
        if refined_score < best_score:
            best_score = refined_score
            best_f = refined_f
    
    # Local refinement: FFT-based sensitivity peak reduction
    if time.time() - start_time < TIME_LIMIT - 30:
        refined_f, refined_score = local_refinement(
            best_f, start_time, TIME_LIMIT - 12
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Second refinement pass with different threshold
    if time.time() - start_time < TIME_LIMIT - 15:
        refined2_f, refined2_score = local_refinement(
            best_f, start_time, TIME_LIMIT - 8,
            alpha_init=0.05, threshold_frac=0.95
        )
        if refined2_score < best_score:
            best_f = refined2_f
            best_score = refined2_score
    
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
