"""Rewritten program for the second autocorrelation inequality task."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
M = 2 * N - 1  # length of g = convolve(h, h)


def compute_value(h):
    """Compute the value metric for given heights."""
    h = np.asarray(h, dtype=float)
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if np.any(h < 0) or not np.all(np.isfinite(h)):
        return 0.0
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    w = 1.0 / (m + 1)
    y_sq_terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq_terms)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)


def neg_value(h):
    """Negative of compute_value for minimization."""
    return -compute_value(h)

def gmm_from_params(params):
    """
    Reconstruct h from GMM parameters.
    params: [amp1, cent1, width1, amp2, cent2, width2, amp3, cent3, width3]
    """
    x = np.linspace(-1, 1, N)
    h = np.zeros(N)
    for i in range(3):
        amp = params[3*i]
        cent = params[3*i + 1]
        width = params[3*i + 2]
        # Ensure width is not too small to avoid numerical issues
        width = max(width, 0.01)
        h += amp * np.exp(-((x - cent) ** 2) / (2 * width ** 2))
    return np.maximum(0, h)

def neg_value_gmm(params):
    """Negative value for GMM parameter optimization."""
    h = gmm_from_params(params)
    return -compute_value(h)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin under 150s

    best_h = np.ones(N)
    best_val = compute_value(best_h)

    # Strategy 1: Parameterized Basis Expansion (Gaussian Mixture Model)
    # This reduces the search space from 50 dimensions to 9 dimensions (3 Gaussians x 3 params)
    
    # Define bounds for GMM parameters
    # amp: [0, 10], cent: [-1, 1], width: [0.05, 1.5]
    bounds_gmm = [
        (0, 10), (-1, 1), (0.05, 1.5),
        (0, 10), (-1, 1), (0.05, 1.5),
        (0, 10), (-1, 1), (0.05, 1.5)
    ]

    # Generate multiple random starting points for GMM parameters
    rng = np.random.default_rng(42)
    num_starts = 50
    
    best_params = None
    best_gmm_val = 0.0

    for i in range(num_starts):
        if time.time() > deadline - 20:
            break
            
        # Random initialization
        params = np.array([
            rng.uniform(0.5, 2.0), rng.uniform(-0.5, 0.5), rng.uniform(0.2, 0.8),
            rng.uniform(0.5, 2.0), rng.uniform(-0.5, 0.5), rng.uniform(0.2, 0.8),
            rng.uniform(0.5, 2.0), rng.uniform(-0.5, 0.5), rng.uniform(0.2, 0.8)
        ])
        
        # Add some structured starts
        if i == 0:
            # Symmetric single bump
            params = np.array([1.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5])
        elif i == 1:
            # Two symmetric bumps
            params = np.array([1.0, -0.3, 0.3, 1.0, 0.3, 0.3, 0.0, 0.0, 0.5])
        elif i == 2:
            # Three bumps
            params = np.array([1.0, -0.4, 0.2, 1.0, 0.0, 0.2, 1.0, 0.4, 0.2])

        try:
            result = minimize(
                neg_value_gmm,
                params,
                method='L-BFGS-B',
                bounds=bounds_gmm,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-9}
            )
            
            h_opt = gmm_from_params(result.x)
            v_opt = compute_value(h_opt)
            
            if v_opt > best_gmm_val:
                best_gmm_val = v_opt
                best_params = result.x.copy()
                if v_opt > best_val:
                    best_val = v_opt
                    best_h = h_opt.copy()
                    
        except Exception:
            pass

    # If GMM found a good solution, use it. Otherwise, fall back to raw height optimization on best_h
    # We can also try to refine best_h with raw L-BFGS-B if time permits
    
    if time.time() < deadline - 5:
        # Refine the best found h (whether from GMM or seed) using raw L-BFGS-B
        try:
            result = minimize(
                neg_value,
                best_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_opt = np.maximum(0, result.x)
            v_opt = compute_value(h_opt)
            if v_opt > best_val:
                best_val = v_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    return best_h.tolist()
