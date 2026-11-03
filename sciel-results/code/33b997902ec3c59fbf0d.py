"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Direct first-order optimization over the 600 step heights using
analytic gradients of the smoothed autoconvolution objective.
This allows the function to take any shape, not restricted to a parametric family.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
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

def compute_grad(f, beta=20.0):
    """
    Compute the gradient of the smoothed objective J(f) = smooth_max(f*f) / (sum(f))^2.
    Returns the gradient vector.
    """
    f = np.asarray(f, dtype=np.float64)
    s = np.sum(f)
    if s < 1e-15:
        return np.zeros_like(f)
    
    conv = np.convolve(f, f, mode='full')
    
    # Softmax weights for the smoothed max
    conv_max = np.max(conv)
    if conv_max < 1e-15:
        return np.zeros_like(f)
    
    shifted = conv - conv_max
    exp_shifted = np.exp(beta * shifted)
    sum_exp = np.sum(exp_shifted)
    w = exp_shifted / sum_exp
    
    # K = smooth_max(conv)
    K = conv_max + (1.0 / beta) * np.log(sum_exp)
    
    # Gradient of K w.r.t f is 2 * (w * f)
    grad_K = 2.0 * np.convolve(w, f, mode='full')[:N]
    
    # Gradient of J = K / s^2
    grad_J = grad_K / (s * s) - (2.0 * K / (s * s * s)) * np.ones(N)
    
    return grad_J

def local_refinement(h, start_time, max_time):
    """Local refinement using sensitivity-based peak reduction."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.05
    threshold_frac = 0.95
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = np.convolve(current_h, current_h, mode='full')
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        threshold = max_c * threshold_frac
        peak_indices = np.where(conv >= threshold)[0]
        
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
        
        conv_new = np.convolve(h_new, h_new, mode='full')
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.05, 0.5)
        else:
            alpha *= 0.85
            if alpha < 1e-8:
                break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    rng = np.random.RandomState(42)
    
    for restart in range(5):
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        
        if restart == 0:
            x = np.linspace(-0.25, 0.25, N, endpoint=False)
            f0 = np.exp(-10 * x**2)
        else:
            f0 = np.abs(rng.normal(0, 1, N)) + 0.1
            f0 = gaussian_filter1d(f0, sigma=5)
        
        f0 = f0 / np.sum(f0)
        
        def objective_and_grad(f):
            f = np.maximum(f, 0.0)
            s = np.sum(f)
            if s < 1e-15:
                return 1e10, np.zeros_like(f)
            conv = np.convolve(f, f, mode='full')
            val = np.max(conv) / (s * s)
            grad = compute_grad(f, beta=20.0)
            return val, grad
        
        try:
            result = minimize(
                objective_and_grad, f0, jac=True,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            
            if result.success or result.fun < best_score:
                f_res = np.maximum(result.x, 0.0)
                s_res = np.sum(f_res)
                if s_res > 0:
                    f_res = f_res / s_res
                score_res = compute_score(f_res)
                if score_res < best_score:
                    best_score = score_res
                    best_f = f_res
        except Exception:
            pass

    if time.time() - start_time < TIME_LIMIT - 10:
        refined_f, refined_score = local_refinement(
            best_f, start_time, TIME_LIMIT - 5
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score

    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
