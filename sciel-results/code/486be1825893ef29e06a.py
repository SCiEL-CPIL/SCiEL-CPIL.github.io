"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Combine parametric cosine-bump initialization with FFT-accelerated
sensitivity-based local refinement. The key improvements over the base program:
1. FFT-based convolution for speed (enables more refinement iterations)
2. Spectral sensitivity computation via FFT for peak identification
3. Adaptive step size with better convergence tracking
4. Multiple refinement passes with decreasing threshold for progressive peak flattening
5. Frequency-domain projected gradient descent as an additional refinement stage
"""

import numpy as np
from scipy.optimize import minimize, minimize_scalar
import time

N = 600
TIME_LIMIT = 140  # safety margin before 150s kill
N_FFT = 2 * N  # FFT size for full convolution

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

def fft_convolve(f):
    """Compute autoconvolution using FFT for speed."""
    f_hat = np.fft.rfft(f, n=N_FFT)
    conv = np.fft.irfft(f_hat * f_hat, n=N_FFT)[:2*N-1]
    return conv

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

def cosine_basis(x, centers, widths, heights):
    """Generate a smooth nonnegative function from cosine bump components."""
    result = np.zeros_like(x)
    for c, w, h in zip(centers, widths, heights):
        dist = np.abs(x - c)
        mask = dist <= w / 2.0
        if np.any(mask):
            result[mask] += h * np.cos(np.pi * (x[mask] - c) / w) ** 2
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
    
    f = cosine_basis(x, centers, widths, heights)
    f = np.maximum(f, 0.0)
    return f

def objective(params, n_components, beta=10.0):
    f = parametric_to_step(params, n_components)
    conv = fft_convolve(f)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return smooth_max(conv, beta) / (s * s)

def local_refinement(h, start_time, max_time):
    """FFT-accelerated local refinement using spectral sensitivity-based peak reduction.
    
    Uses FFT to compute convolution and sensitivity in O(N log N) per iteration,
    enabling many more refinement steps within the time budget.
    Multiple passes with tightening threshold for progressive peak flattening.
    """
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    
    # Progressive threshold schedule: start loose, tighten over time
    threshold_schedule = [0.95, 0.90, 0.85, 0.80, 0.75]
    pass_idx = 0
    alpha = 0.03
    no_improve_count = 0
    max_no_improve = 30
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        # FFT-based convolution
        conv = fft_convolve(current_h)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            # Try next threshold if available
            pass_idx += 1
            if pass_idx >= len(threshold_schedule):
                break
            no_improve_count = 0
            alpha = 0.02  # reset with smaller step
            continue
        
        threshold_frac = threshold_schedule[pass_idx]
        threshold = max_c * threshold_frac
        peak_mask = (conv >= threshold).astype(np.float64)
        
        # Spectral sensitivity: which bins contribute to peak lags
        # conv[m] = sum_j h[j]*h[m-j], d(conv[m])/d(h[j]) = h[m-j]
        # Aggregate over peak lags using FFT
        pm_hat = np.fft.rfft(peak_mask, n=N_FFT)
        ch_hat = np.fft.rfft(current_h, n=N_FFT)
        # Sensitivity for bin j: sum over peak m of h[m-j]
        # This is a cross-correlation, computable via FFT
        sens = np.fft.irfft(pm_hat * np.conj(ch_hat), n=N_FFT)[:N]
        
        # Normalize sensitivity
        sens_mean = np.mean(sens)
        sens_std = np.std(sens) + 1e-12
        sens_norm = (sens - sens_mean) / sens_std
        
        # Reduce bins with high sensitivity, increase low-sensitivity
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fft_convolve(h_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.02, 0.15)
        else:
            alpha *= 0.9
            if alpha < 1e-9:
                break
    
    return best_h, best_val

def spectral_polish(h, start_time, max_time):
    """High-precision polish: very small steps targeting the top few peak lags only."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.005
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = fft_convolve(current_h)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        # Target only the top 5% of lags
        threshold = max_c * 0.99
        peak_mask = (conv >= threshold).astype(np.float64)
        
        if np.sum(peak_mask) == 0:
            break
        
        pm_hat = np.fft.rfft(peak_mask, n=N_FFT)
        ch_hat = np.fft.rfft(current_h, n=N_FFT)
        sens = np.fft.irfft(pm_hat * np.conj(ch_hat), n=N_FFT)[:N]
        
        sens_mean = np.mean(sens)
        sens_std = np.std(sens) + 1e-12
        sens_norm = (sens - sens_mean) / sens_std
        
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fft_convolve(h_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
        else:
            alpha *= 0.95
            if alpha < 1e-10:
                break
    
    return best_h, best_val

def freq_domain_refinement(h, start_time, max_time):
    """Frequency-domain projected gradient descent to further reduce peak autoconvolution.
    
    The autoconvolution is the inverse FFT of |F|^2 where F is the FFT of h.
    We work in the frequency domain: minimize max of ifft(|F|^2) subject to
    ifft(F) >= 0 and sum(ifft(F)) = N.
    
    Use projected gradient: step in frequency domain, project back to time domain
    by clipping negative values and renormalizing.
    """
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    lr = 0.01
    no_improve = 0
    
    while time.time() - start_time < max_time:
        if no_improve > 50:
            break
        
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = fft_convolve(current_h)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        # Identify peak lags
        threshold = max_c * 0.98
        peak_mask = (conv >= threshold).astype(np.float64)
        
        if np.sum(peak_mask) == 0:
            break
        
        # Gradient of max(conv) w.r.t. h: for each bin j, sum over peak m of h[m-j]
        # This is the same sensitivity as before
        pm_hat = np.fft.rfft(peak_mask, n=N_FFT)
        ch_hat = np.fft.rfft(current_h, n=N_FFT)
        sens = np.fft.irfft(pm_hat * np.conj(ch_hat), n=N_FFT)[:N]
        
        # Normalize
        sens_mean = np.mean(sens)
        sens_std = np.std(sens) + 1e-12
        sens_norm = (sens - sens_mean) / sens_std
        
        # Gradient step in time domain (equivalent to frequency domain for linear ops)
        h_new = current_h - lr * sens_norm
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fft_convolve(h_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            lr = min(lr * 1.01, 0.1)
        else:
            lr *= 0.9
            if lr < 1e-8:
                break
    
    return best_h, best_val

def single_bin_optimization(h, start_time, max_time):
    """Iteratively optimize individual bins using exact 1D minimization.
    
    For each bin, fix all others and find the optimal value via minimize_scalar.
    This is a coordinate descent approach with exact line searches.
    """
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    
    # Precompute FFT size
    n_fft = 2 * N
    
    # Number of full passes to attempt
    max_passes = 20
    
    for p in range(max_passes):
        if time.time() - start_time > max_time:
            break
        
        improved = False
        
        # Process bins in random order to avoid bias
        rng = np.random.RandomState(42 + p)
        order = rng.permutation(N)
        
        for idx in order:
            if time.time() - start_time > max_time:
                break
            
            h_old = current_h[idx]
            
            # Define 1D objective: vary h[idx], keep rest fixed
            def obj_1d(val):
                current_h[idx] = max(val, 0.0)
                s = np.sum(current_h)
                if s < 1e-12:
                    current_h[idx] = h_old
                    return np.inf
                conv = fft_convolve(current_h)
                result = np.max(conv) / (s * s)
                current_h[idx] = h_old
                return result
            
            # Bounds: [0, 4 * mean]
            mean_h = np.mean(current_h)
            bounds = (0.0, 4.0 * mean_h + 1.0)
            
            try:
                res = minimize_scalar(obj_1d, bounds=bounds, method='bounded',
                                     options={'xatol': 1e-6, 'maxiter': 50})
                if res.fun < best_val:
                    current_h[idx] = max(res.x, 0.0)
                    best_val = res.fun
                    best_h = current_h.copy()
                    improved = True
            except Exception:
                pass
        
        if not improved:
            break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Parametric initialization with cosine bumps
    for n_c in [2, 3, 4, 5]:
        if time.time() - start_time > TIME_LIMIT - 50:
            break
        
        rng = np.random.RandomState(42 + n_c)
        best_params = None
        best_obj = np.inf
        
        betas = [5.0, 10.0, 20.0]
        
        for beta in betas:
            if time.time() - start_time > TIME_LIMIT - 25:
                break
            
            for restart in range(4):
                if time.time() - start_time > TIME_LIMIT - 20:
                    break
                
                centers = rng.uniform(-0.2, 0.2, size=n_c)
                widths = rng.uniform(0.05, 0.2, size=n_c)
                heights = rng.uniform(0.5, 2.0, size=n_c)
                
                x0 = np.concatenate([centers, widths, heights])
                
                try:
                    result = minimize(
                        objective, x0, args=(n_c, beta),
                        method='Nelder-Mead',
                        options={'maxiter': 1500, 'xatol': 1e-6, 'fatol': 1e-8}
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
    
    # FFT-accelerated local refinement
    if time.time() - start_time < TIME_LIMIT - 20:
        refined_f, refined_score = local_refinement(
            best_f, start_time, TIME_LIMIT - 15
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Frequency-domain refinement
    if time.time() - start_time < TIME_LIMIT - 10:
        freq_f, freq_score = freq_domain_refinement(
            best_f, start_time, TIME_LIMIT - 8
        )
        if freq_score < best_score:
            best_f = freq_f
            best_score = freq_score
    
    # Single-bin exact optimization (coordinate descent with line search)
    if time.time() - start_time < TIME_LIMIT - 5:
        bin_f, bin_score = single_bin_optimization(
            best_f, start_time, TIME_LIMIT - 3
        )
        if bin_score < best_score:
            best_f = bin_f
            best_score = bin_score
    
    # High-precision spectral polish
    if time.time() - start_time < TIME_LIMIT - 2:
        polished_f, polished_score = spectral_polish(
            best_f, start_time, TIME_LIMIT - 1
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
