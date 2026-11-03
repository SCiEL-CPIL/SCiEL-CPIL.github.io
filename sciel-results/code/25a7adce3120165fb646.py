import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 50
w = 1.0 / N  # interval width
half_range = 1.0 / 4

def compute_ratio(h):
    """Compute the ratio L2^2 / (L1 * sup|autoconv|) for step heights h."""
    h = np.maximum(h, 0)  # ensure nonnegative
    if np.all(h == 0):
        return 0.0
    
    L2_sq = np.sum(h**2) * w
    L1 = np.sum(h) * w
    
    if L1 == 0:
        return 0.0
    
    # Compute autoconvolution of step function
    # For step function with heights h_i on intervals [x_i, x_i + w]
    # The autoconvolution at point x is integral of f(t)f(x-t)dt
    # For discrete steps, we can compute it numerically
    
    # Positions of interval centers
    x = np.linspace(-half_range + w/2, half_range - w/2, N)
    
    # Autoconvolution: (f*f)(y) = sum_i sum_j h_i * h_j * length of intersection
    # of [x_i, x_i+w] and [y-x_j-w, y-x_j]
    # This is complex; let's use FFT-based convolution
    
    # Pad and use FFT for convolution
    # f is represented as step function, so f(x) = h_i for x in [x_i - w/2, x_i + w/2]
    # Autoconvolution = f * f
    
    # Use numerical integration approach
    # Sample the autoconvolution on a fine grid
    n_samples = 512
    y = np.linspace(-2*half_range, 2*half_range, n_samples)
    
    # f(x) for any x: find which interval x falls in
    def f_val(x):
        # x is array
        # interval i covers [x_i - w/2, x_i + w/2] where x_i = -half_range + (i+0.5)*w
        # So interval i: [-half_range + i*w, -half_range + (i+1)*w]
        idx = ((x + half_range) / w).astype(int)
        idx = np.clip(idx, 0, N-1)
        return h[idx]
    
    fv = f_val(y)
    # Autoconvolution at y: integral of f(t) f(y-t) dt
    # Use trapezoidal rule on a fine grid
    dt = y[1] - y[0]
    # f(y-t) = f_val(y - t) where t = y
    # So (f*f)(y) = integral f(t) f(y-t) dt
    # = sum over t: f(t) * f(y-t) * dt
    
    # For each y_k, compute sum_i f(t_i) * f(y_k - t_i) * dt
    # This is O(n_samples^2) which is 512^2 = 262144, fine
    
    # Precompute f on grid
    # f(y_k - t_i) for all k, i
    # t_i = y_i, so y_k - t_i = y_k - y_i
    # We need f at points y_k - y_i
    
    # Create meshgrid
    Y, T = np.meshgrid(y, y, indexing='ij')
    FT = f_val(Y - T)
    autoconv = np.sum(fv[np.newaxis, :] * FT, axis=1) * dt
    
    sup_autoconv = np.max(np.abs(autoconv))
    
    if sup_autoconv == 0:
        return 0.0
    
    return L2_sq / (L1 * sup_autoconv)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    # Try different initial guesses and optimize
    best_h = np.ones(N)
    best_score = compute_ratio(best_h)
    
    # Try several random initializations with differential evolution or Nelder-Mead
    rng = np.random.default_rng(42)
    
    for trial in range(20):
        if time.time() > deadline - 5.0:
            break
        
        # Different initial patterns
        if trial == 0:
            h0 = np.ones(N)
        elif trial == 1:
            # Gaussian-like
            x = np.linspace(-1, 1, N)
            h0 = np.exp(-x**2 * 2)
        elif trial == 2:
            # Cosine bump
            x = np.linspace(-1, 1, N)
            h0 = np.maximum(0, np.cos(x * np.pi / 2))
        elif trial == 3:
            # Triangular
            x = np.linspace(-1, 1, N)
            h0 = np.maximum(0, 1 - np.abs(x))
        elif trial == 4:
            # Flat top
            h0 = np.ones(N)
            h0[:10] = 0
            h0[-10:] = 0
        else:
            h0 = rng.random(N)
        
        # Use Nelder-Mead for optimization
        try:
            result = minimize(
                lambda h: -compute_ratio(h),
                h0,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            h_opt = np.maximum(result.x, 0)
            score_opt = compute_ratio(h_opt)
            
            if score_opt > best_score:
                best_score = score_opt
                best_h = h_opt
        except Exception:
            pass
    
    # Final polish with more iterations if time permits
    if time.time() < deadline - 10.0:
        try:
            result = minimize(
                lambda h: -compute_ratio(h),
                best_h,
                method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            h_opt = np.maximum(result.x, 0)
            score_opt = compute_ratio(h_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_h = h_opt
        except Exception:
            pass
    
    return list(best_h)
