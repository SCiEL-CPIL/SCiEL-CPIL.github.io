"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Use a smooth parametric family based on cosine bumps, then refine
with a sensitivity-based local optimization that targets peak reduction.

The key insight is to use a family of functions whose autoconvolution has a
flatter peak. We use a smooth max (log-sum-exp) surrogate for the parametric
optimization, then apply a gradient-like local refinement that reduces the
sensitivity of peak lags to reduce the maximum autoconvolution value.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 130  # safety margin before 150s kill

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

def fft_autocorr(f):
    """Compute autoconvolution using FFT for speed."""
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    F = np.fft.fft(f, size)
    conv = np.fft.ifft(F * F).real[:2 * n - 1]
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
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return smooth_max(conv, beta) / (s * s)

def local_refinement(h, start_time, max_time):
    """Local refinement using coordinate descent with exact 1D minimization."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    
    # Normalize to sum = 1 for simplicity in objective scaling, or keep N
    # The score is max(conv) / (sum(f)^2). Scaling f by k scales conv by k^2 and sum^2 by k^2.
    # So the score is invariant to scaling. We can fix sum(f) = 1 or N.
    # Let's fix sum(f) = 1 to keep numbers small.
    s = np.sum(current_h)
    if s > 0:
        current_h = current_h / s
    
    def obj(f):
        conv = fft_autocorr(f)
        return np.max(conv)
        
    # Coordinate Descent
    # We iterate through bins, and for each bin, we minimize the objective with respect to that bin.
    # Since sum(f) is fixed, changing f[i] requires adjusting others? No, usually we just change f[i] and let the scale be handled or we optimize over f[i] with the constraint that f[i] >= 0.
    # Wait, if we fix sum(f)=1, changing f[i] breaks the sum constraint unless we adjust others.
    # Alternative: Don't fix sum. The objective is homogeneous of degree 0.
    # So we can just minimize max(conv) subject to f >= 0.
    # But if we scale f up, max(conv) goes up. So we must normalize.
    # Let's define the objective as max(conv) / (sum(f)^2).
    
    def obj_normalized(f):
        s = np.sum(f)
        if s < 1e-15:
            return np.inf
        conv = fft_autocorr(f)
        return np.max(conv) / (s * s)

    iterations = 0
    while time.time() - start_time < max_time - 5:
        improved = False
        for i in range(N):
            if time.time() - start_time > max_time - 2:
                break
            
            # Save current value
            current_val = obj_normalized(current_h)
            
            # Define 1D function for bin i
            def f_1d(x_i):
                f_new = current_h.copy()
                f_new[i] = x_i
                return obj_normalized(f_new)
            
            # Bounds: [0, max_val * 2]
            max_val = np.max(current_h)
            if max_val == 0:
                max_val = 1.0
            bounds = (0, max_val * 2)
            
            try:
                res = minimize_scalar(f_1d, bounds=bounds, method='bounded', options={'xatol': 1e-6})
                if res.success and res.fun < current_val - 1e-12:
                    current_h[i] = res.x
                    improved = True
            except Exception:
                pass
        
        # Update best
        current_val = obj_normalized(current_h)
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        if not improved:
            break
            
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Try different numbers of cosine bump components
    for n_c in [2, 3, 4, 5]:
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        
        rng = np.random.RandomState(42 + n_c)
        best_params = None
        best_obj = np.inf
        
        # Annealed beta: start lower, increase for sharper max approximation
        betas = [5.0, 10.0, 20.0]
        
        for beta in betas:
            if time.time() - start_time > TIME_LIMIT - 15:
                break
            
            for restart in range(6):
                if time.time() - start_time > TIME_LIMIT - 10:
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
    
    # Local refinement: Coordinate Descent
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
