"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. Use a smooth parametric family (cosine bumps) to get a good initial guess.
2. Use FFT-based autoconvolution for fast evaluation.
3. Apply a sensitivity-based local refinement that uses exact 1D minimization
   (scipy.optimize.minimize_scalar) on individual bins to reduce the peak.
4. Use a smooth max surrogate during parametric optimization.
"""

import numpy as np
from scipy.optimize import minimize, minimize_scalar
import time

N = 600
TIME_LIMIT = 135  # safety margin before 150s kill

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

def fft_autoconv(f):
    """Compute autoconvolution using FFT for speed."""
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    conv = np.fft.irfft(F * F, n=n_fft)[:2*N-1]
    return conv

def smooth_max(x, beta):
    """Smooth approximation to max using log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0
    shifted = x - x_max
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
    conv = fft_autoconv(f)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return smooth_max(conv, beta) / (s * s)

def local_refinement(h, start_time, max_time):
    """Local refinement using sensitivity-based peak reduction with exact 1D minimization.
    
    At each step, we identify lags where the autoconvolution is near its max,
    compute how much each bin contributes to those peaks, and use exact 1D
    minimization to adjust the most impactful bins.
    """
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    
    # Normalize to sum = N for stable computation
    s = np.sum(current_h)
    if s > 1e-12:
        current_h = current_h * (N / s)
    
    threshold_frac = 0.90
    max_iterations = 500
    iteration = 0
    
    while time.time() - start_time < max_time and iteration < max_iterations:
        iteration += 1
        s = np.sum(current_h)
        if s < 1e-12:
            break
        
        # Normalize to sum = N
        current_h = current_h * (N / s)
        
        conv = fft_autoconv(current_h)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        # Find lags where conv is close to max
        threshold = max_c * threshold_frac
        peak_indices = np.where(conv >= threshold)[0]
        
        # Compute sensitivity: for each bin j, how much does it contribute to peak lags?
        # conv[m] = sum_j current_h[j] * current_h[m-j]
        # d(conv[m])/d(current_h[j]) = current_h[m-j]
        sensitivity = np.zeros(N)
        for m in peak_indices:
            # m ranges from 0 to 2N-2
            # j ranges such that m-j is in [0, N-1]
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            for j in range(j_min, j_max + 1):
                sensitivity[j] += current_h[m - j]
        
        # Normalize sensitivity
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        # Select top K bins with highest sensitivity to adjust
        K = 20
        top_indices = np.argsort(sensitivity)[-K:]
        
        # For each selected bin, perform exact 1D minimization
        improved = False
        for j in top_indices:
            if time.time() - start_time > max_time:
                break
            
            h_j = current_h[j]
            
            # Define 1D objective: adjust h[j] and renormalize
            def obj_1d(delta):
                h_new = current_h.copy()
                h_new[j] = max(0.0, h_j + delta)
                s_new = np.sum(h_new)
                if s_new < 1e-12:
                    return np.inf
                h_new = h_new * (N / s_new)
                conv_new = fft_autoconv(h_new)
                return np.max(conv_new) / (N * N)
            
            # Minimize over delta in a reasonable range
            result = minimize_scalar(
                obj_1d,
                bounds=(-h_j, 0.5 * h_j + 0.1),
                method='bounded',
                options={'xatol': 1e-8}
            )
            
            if result.fun < current_val:
                h_new = current_h.copy()
                h_new[j] = max(0.0, h_j + result.x)
                s_new = np.sum(h_new)
                if s_new > 1e-12:
                    h_new = h_new * (N / s_new)
                    current_h = h_new
                    improved = True
        
        if not improved:
            # Try a smaller step size approach
            alpha = 0.02
            h_new = current_h * (1.0 - alpha * sens_norm)
            h_new = np.maximum(h_new, 0.0)
            s_new = np.sum(h_new)
            if s_new > 1e-12:
                h_new = h_new * (N / s_new)
                conv_new = fft_autoconv(h_new)
                val_new = np.max(conv_new) / (N * N)
                if val_new < current_val:
                    current_h = h_new
                else:
                    # No improvement, break
                    break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Try different numbers of cosine bump components
    for n_c in [2, 3, 4, 5]:
        if time.time() - start_time > TIME_LIMIT - 40:
            break
        
        rng = np.random.RandomState(42 + n_c)
        best_params = None
        best_obj = np.inf
        
        # Annealed beta: start lower, increase for sharper max approximation
        betas = [5.0, 10.0, 20.0]
        
        for beta in betas:
            if time.time() - start_time > TIME_LIMIT - 25:
                break
            
            for restart in range(6):
                if time.time() - start_time > TIME_LIMIT - 15:
                    break
                
                centers = rng.uniform(-0.2, 0.2, size=n_c)
                widths = rng.uniform(0.05, 0.2, size=n_c)
                heights = rng.uniform(0.5, 2.0, size=n_c)
                
                x0 = np.concatenate([centers, widths, heights])
                
                try:
                    result = minimize(
                        objective, x0, args=(n_c, beta),
                        method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
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
    
    # Local refinement: sensitivity-based peak reduction with exact 1D minimization
    if time.time() - start_time < TIME_LIMIT - 10:
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
