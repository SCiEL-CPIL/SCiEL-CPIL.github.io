"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Use a low-dimensional parametric family based on Gaussian bumps,
then refine with a spectral-domain coordinate descent that directly targets
peak reduction in the autoconvolution.

The key insight is to:
1. Start with a smooth parametric family (Gaussian bumps) to find a good
   initial shape.
2. Use spectral-domain optimization to directly minimize the peak of the
   autoconvolution by adjusting Fourier coefficients.
3. Apply a final polish pass with tight tolerance.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 145  # safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
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
        result += h * np.exp(-0.5 * ((x - c) / w) ** 2)
    return result

def cosine_basis(x, centers, widths, heights):
    """Generate a smooth nonnegative function from cosine bump components."""
    result = np.zeros_like(x)
    for c, w, h in zip(centers, widths, heights):
        u = (x - c) / w
        mask = np.abs(u) < 1.0
        result[mask] += h * np.cos(np.pi * u[mask]) ** 2
    return result

def parametric_to_step(params, n_components, use_cosine=False):
    """Convert parameters to a step function of length N."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    centers = params[:n_components]
    widths = params[n_components:2*n_components]
    heights = params[2*n_components:3*n_components]
    
    widths = np.abs(widths) + 0.02
    widths = np.minimum(widths, 0.5)
    heights = np.abs(heights)
    
    if use_cosine:
        f = cosine_basis(x, centers, widths, heights)
    else:
        f = gaussian_basis(x, centers, widths, heights)
    f = np.maximum(f, 0.0)
    return f

def objective(params, n_components, beta=10.0, use_cosine=False):
    f = parametric_to_step(params, n_components, use_cosine=use_cosine)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return smooth_max(conv, beta) / (s * s)

def spectral_refinement(h, start_time, max_time):
    """Fejes-Tóth spectral decomposition: optimize autoconvolution directly.
    
    The autoconvolution c[k] = sum_i f[i]*f[k-i] is a nonnegative trigonometric
    polynomial. We work in the frequency domain where the autoconvolution's
    DFT is |F|^2 (real and nonnegative). We flatten the time-domain peak by
    adjusting the magnitude spectrum, then project back to ensure f >= 0.
    """
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    n_fft = 2 * N
    n_freq = n_fft // 2 + 1
    
    # Initialize with current spectrum
    F_mag = np.abs(np.fft.rfft(current_h, n=n_fft))
    F_phase = np.angle(np.fft.rfft(current_h, n=n_fft))
    
    alpha = 0.01
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        # Compute current autoconvolution
        f_hat = np.fft.rfft(current_h, n=n_fft)
        conv = np.fft.irfft(f_hat * f_hat, n=n_fft)[:2*N-1]
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            break
        
        # Find peak region
        peak_idx = np.argmax(conv)
        
        # In frequency domain, the autoconvolution's DFT is |F|^2
        # We want to reduce energy at frequencies that contribute to the peak
        # The sensitivity of conv[peak] to |F[k]|^2 is related to the Fourier
        # transform of the delta at peak, which is a complex exponential
        
        # Compute gradient in spectral domain
        # d(conv[peak])/d(|F[k]|^2) = Re{exp(-2*pi*i*peak*k/n_fft)}
        # But we need to be careful: conv is the IFFT of |F|^2
        
        # Actually, let's work directly: 
        # conv = IFFT(|F|^2)
        # d(conv[peak])/d(F_mag[k]^2) = Re{IFFT basis at peak}
        
        # The IFFT of a single frequency component k is:
        # (1/n_fft) * exp(2*pi*i*k*peak/n_fft)
        # So d(conv[peak])/d(|F[k]|^2) = (1/n_fft) * cos(2*pi*k*peak/n_fft)
        
        k_vals = np.arange(n_freq)
        # For real signals, we need to account for conjugate symmetry
        # The contribution to conv[peak] from |F[k]|^2 is:
        # (1/n_fft) * |F[k]|^2 * cos(2*pi*k*peak/n_fft) for k=0, n_fft/2
        # (2/n_fft) * |F[k]|^2 * cos(2*pi*k*peak/n_fft) for 0 < k < n_fft/2
        
        grad_spec = np.zeros(n_freq)
        for k in range(n_freq):
            if k == 0 or k == n_fft // 2:
                grad_spec[k] = (1.0 / n_fft) * np.cos(2.0 * np.pi * k * peak_idx / n_fft)
            else:
                grad_spec[k] = (2.0 / n_fft) * np.cos(2.0 * np.pi * k * peak_idx / n_fft)
        
        # Normalize
        grad_norm = np.linalg.norm(grad_spec)
        if grad_norm < 1e-12:
            break
        grad_spec = grad_spec / grad_norm
        
        # Update magnitude spectrum: reduce components that increase the peak
        # Move in direction that decreases conv[peak]
        F_mag_new = F_mag - alpha * grad_spec * np.std(F_mag)
        F_mag_new = np.maximum(F_mag_new, 1e-10)  # Keep positive
        
        # Reconstruct time-domain signal
        F_new = F_mag_new * np.exp(1j * F_phase)
        h_new = np.fft.irfft(F_new, n=n_fft)[:N]
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        # Update spectrum to match new time-domain signal
        f_new_hat = np.fft.rfft(h_new, n=n_fft)
        F_mag = np.abs(f_new_hat)
        F_phase = np.angle(f_new_hat)
        
        # Evaluate
        conv_new = np.fft.irfft(f_new_hat * f_new_hat, n=n_fft)[:2*N-1]
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.02, 0.05)
        else:
            alpha *= 0.9
            if alpha < 1e-8:
                break
    
    return best_h, best_val

def final_polish(h, start_time, max_time):
    """Final high-precision polish using smooth max surrogate in spectral domain."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    n_fft = 2 * N
    n_freq = n_fft // 2 + 1
    alpha = 0.003
    no_improve_count = 0
    max_no_improve = 80
    beta = 20.0  # Smooth max parameter
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        f_hat = np.fft.rfft(current_h, n=n_fft)
        conv = np.fft.irfft(f_hat * f_hat, n=n_fft)[:2*N-1]
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            break
        
        # Use smooth max surrogate for differentiability
        # smooth_max(conv) = (1/beta) * log(sum(exp(beta * conv / max_c))) * max_c
        # Gradient w.r.t. conv[k] = (exp(beta*conv[k]/max_c) / sum(exp(...)))
        
        conv_scaled = conv / (max_c + 1e-15)
        exp_conv = np.exp(beta * conv_scaled)
        weights = exp_conv / (np.sum(exp_conv) + 1e-15)
        
        # The gradient of smooth_max w.r.t. f[i] is:
        # sum_k weights[k] * d(conv[k])/df[i]
        # d(conv[k])/df[i] = f[k-i] (with boundary handling)
        
        # Compute via FFT for efficiency
        # weights_hat = FFT(weights)
        # grad = IFFT(weights_hat * conj(f_hat))
        
        w_hat = np.fft.rfft(weights, n=n_fft)
        grad = np.fft.irfft(w_hat * np.conj(f_hat), n=n_fft)[:N]
        
        # Normalize
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        grad = grad / grad_norm
        
        # Apply update
        h_new = current_h - alpha * grad * np.std(current_h)
        h_new = np.maximum(h_new, 0.0)
        
        # If too many zeros, blend back
        zero_frac = np.mean(h_new < 1e-10)
        if zero_frac > 0.4:
            h_new = 0.3 * h_new + 0.7 * current_h
            h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        f_new_hat = np.fft.rfft(h_new, n=n_fft)
        conv_new = np.fft.irfft(f_new_hat * f_new_hat, n=n_fft)[:2*N-1]
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.015, 0.015)
        else:
            alpha *= 0.92
            if alpha < 1e-9:
                break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Try different numbers of bump components with both Gaussian and cosine bases
    param_budget_end = start_time + (TIME_LIMIT - 5) * 0.25
    
    for use_cosine in [False, True]:
        if time.time() > param_budget_end:
            break
            
        for n_c in [2, 3, 4, 5]:
            if time.time() > param_budget_end:
                break
            
            rng = np.random.RandomState(42 + n_c + (100 if use_cosine else 0))
            best_params = None
            best_obj = np.inf
            
            betas = [10.0, 20.0, 50.0]
            
            for beta in betas:
                if time.time() > param_budget_end:
                    break
                
                for restart in range(4):
                    if time.time() > param_budget_end:
                        break
                    
                    centers = rng.uniform(-0.15, 0.15, size=n_c)
                    widths = rng.uniform(0.05, 0.20, size=n_c)
                    heights = rng.uniform(0.5, 3.0, size=n_c)
                    
                    x0 = np.concatenate([centers, widths, heights])
                    
                    try:
                        result = minimize(
                            objective, x0, args=(n_c, beta, use_cosine),
                            method='Nelder-Mead',
                            options={'maxiter': 600, 'xatol': 1e-6, 'fatol': 1e-8}
                        )
                        if result.fun < best_obj:
                            best_obj = result.fun
                            best_params = result.x
                    except Exception:
                        pass
            
            if best_params is not None:
                f = parametric_to_step(best_params, n_c, use_cosine=use_cosine)
                actual_score = compute_score(f)
                if actual_score < best_score:
                    best_f = f
                    best_score = actual_score
    
    # Spectral refinement - Fejes-Tóth decomposition
    if time.time() < TIME_LIMIT - 15:
        refined_f, refined_score = spectral_refinement(
            best_f, start_time, TIME_LIMIT - 15
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Final polish with smooth max surrogate
    if time.time() < TIME_LIMIT - 3:
        polished_f, polished_score = final_polish(
            best_f, start_time, TIME_LIMIT - 3
        )
        if polished_score < best_score:
            best_f = polished_f
            best_score = polished_score
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    
    # Final normalization to ensure sum = N
    s_final = np.sum(best_f)
    if s_final > 1e-12:
        best_f = best_f * (N / s_final)
    
    # One last quick polish with very small steps using smooth max
    if time.time() < TIME_LIMIT - 1:
        for _ in range(30):
            if time.time() > TIME_LIMIT - 0.5:
                break
            f_hat = np.fft.rfft(best_f, n=2*N)
            conv = np.fft.irfft(f_hat * f_hat, n=2*N)[:2*N-1]
            max_c = np.max(conv)
            s = np.sum(best_f)
            if s < 1e-12 or max_c < 1e-15:
                break
            
            # Smooth max gradient
            conv_scaled = conv / max_c
            exp_conv = np.exp(20.0 * conv_scaled)
            weights = exp_conv / (np.sum(exp_conv) + 1e-15)
            
            # Compute gradient via FFT
            w_hat = np.fft.rfft(weights, n=2*N)
            grad = np.fft.irfft(w_hat * np.conj(f_hat), n=2*N)[:N]
            
            gn = np.linalg.norm(grad)
            if gn < 1e-12:
                break
            grad = grad / gn
            
            h_new = best_f - 0.002 * grad * np.std(best_f)
            h_new = np.maximum(h_new, 0.0)
            s_new = np.sum(h_new)
            if s_new < 1e-12:
                break
            h_new = h_new * (N / s_new)
            
            fh = np.fft.rfft(h_new, n=2*N)
            cn = np.fft.irfft(fh * fh, n=2*N)[:2*N-1]
            if np.max(cn) / (N*N) < compute_score(best_f):
                best_f = h_new
    
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
