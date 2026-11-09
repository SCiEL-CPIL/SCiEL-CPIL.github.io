import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _compute_metric(h):
    """Compute the upper bound on C3: max abs autocorrelation at non-zero lags / N."""
    H = np.fft.fft(h)
    R = np.fft.ifft(np.abs(H) ** 2).real
    max_abs = np.max(np.abs(R[1:]))
    return max_abs / N

def _smooth_objective(x, sharpness=50.0):
    """
    Smooth surrogate for the peak autocorrelation.
    x is a continuous vector of length N.
    We map x to signs via tanh(sharpness * x), then compute the peak autocorrelation.
    To make it differentiable, we use a soft-max approximation of the max.
    """
    signs = np.tanh(sharpness * x)
    H = np.fft.fft(signs)
    R = np.fft.ifft(np.abs(H) ** 2).real
    # Use a soft-max of the absolute values at non-zero lags
    abs_R = np.abs(R[1:])
    # Soft max with temperature
    temp = 0.5
    soft_max = temp * np.log(np.sum(np.exp(abs_R / temp)))
    return soft_max / N

def _smooth_gradient(x, sharpness=50.0):
    """
    Numerical gradient of the smooth objective.
    """
    eps = 1e-7
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        grad[i] = (_smooth_objective(x_plus, sharpness) - _smooth_objective(x_minus, sharpness)) / (2 * eps)
    return grad

def _greedy_polish(h, deadline):
    """Greedy polish: try each position, keep if it improves."""
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for idx in range(N):
            if time.time() > deadline - 5:
                break
            h_trial = h.copy()
            h_trial[idx] = -h_trial[idx]
            if _compute_metric(h_trial) < _compute_metric(h) - 1e-12:
                h = h_trial
                improved = True
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    # Try different initializations for the continuous optimizer
    initializations = [
        np.zeros(N),
        np.random.randn(N) * 0.5,
        np.random.randn(N) * 0.1,
    ]
    
    for init in initializations:
        if time.time() > deadline - 60:
            break
        
        x0 = init.copy()
        
        # Run L-BFGS-B on the continuous surrogate
        try:
            result = minimize(
                _smooth_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 50}
            )
            x_opt = result.x
        except Exception:
            continue
        
        # Convert continuous solution to discrete signs
        h_candidate = np.sign(np.tanh(50.0 * x_opt))
        # Ensure no zeros
        h_candidate[h_candidate == 0] = 1.0
        
        score = _compute_metric(h_candidate)
        if score < best_score:
            best_score = score
            best_h = h_candidate.copy()
    
    # If we haven't found anything yet, start with random
    if best_h is None:
        best_h = np.random.choice([-1.0, 1.0], N)
        best_score = _compute_metric(best_h)
    
    # Greedy polish to refine the discrete solution
    if time.time() < deadline - 10:
        best_h = _greedy_polish(best_h, deadline)
    
    return best_h
