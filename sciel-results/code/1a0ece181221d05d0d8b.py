import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _compute_overlaps(h):
    """Compute overlap(k) for all k using FFT."""
    g = 1.0 - h
    H = np.fft.fft(h)
    G = np.fft.fft(g)
    # Cross-correlation: sum_i h[i] * g[(i+k) mod N]
    # This is IFFT(FFT(h) * conj(FFT(g)))
    corr = np.fft.ifft(H * np.conj(G)).real
    return corr

def _max_overlap(h):
    """Maximum overlap value."""
    overlaps = _compute_overlaps(h)
    return np.max(overlaps)

def _objective(x):
    """Objective with penalty for sum constraint."""
    h = 0.5 + x
    # Penalty for sum constraint
    sum_h = np.sum(h)
    penalty = 1e6 * (sum_h - N/2)**2
    # Smooth max using log-sum-exp
    overlaps = _compute_overlaps(h)
    # Clip to avoid numerical issues
    overlaps = np.clip(overlaps, 1e-10, None)
    log_sum_exp = np.log(np.sum(np.exp(overlaps * 100))) / 100
    return log_sum_exp + penalty

def _objective_grad(x):
    h = 0.5 + x
    overlaps = _compute_overlaps(h)
    # Gradient of log-sum-exp
    exp_overlaps = np.exp(overlaps * 100)
    weights = exp_overlaps / np.sum(exp_overlaps)
    # Gradient of overlaps w.r.t. h
    # overlap(k) = sum_i h[i] * (1 - h[(i+k) mod N])
    # d overlap(k) / d h[j] = (1 - h[(j+k) mod N]) + h[j] * (-1) * delta_{j, (i+k)} for i such that i+k=j mod N
    # Actually, let me think more carefully.
    # overlap(k) = sum_i h[i] * (1 - h[(i+k) mod N])
    # d overlap(k) / d h[j] = sum_i [delta_{i,j} * (1 - h[(i+k) mod N]) + h[i] * (-delta_{(i+k) mod N, j})]
    # = (1 - h[(j+k) mod N]) + sum_i h[i] * (-delta_{(i+k) mod N, j})
    # = (1 - h[(j+k) mod N]) - h[(j-k) mod N]
    
    # So gradient of overlap(k) w.r.t. h is: (1 - h[(j+k) mod N]) - h[(j-k) mod N] for each j
    # This can be computed using FFT or directly.
    
    # Actually, let's use the FFT approach for the gradient too.
    # overlap(k) = IFFT(FFT(h) * conj(FFT(1-h))) [k]
    # d overlap / d h = ... this gets complex. Let me just do it directly for N=100.
    
    grad_h = np.zeros(N)
    for k in range(N):
        w = weights[k]
        for j in range(N):
            grad_h[j] += w * ((1 - h[(j+k) % N]) - h[(j-k) % N])
    
    # Gradient of penalty
    penalty_grad = 2 * 1e6 * (np.sum(h) - N/2)
    grad_x = grad_h + penalty_grad
    return grad_x

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    best_h = np.full(N, 0.5)
    best_val = _max_overlap(best_h)
    
    rng = np.random.default_rng(42)
    
    # Try multiple random starts
    n_starts = 50
    for s in range(n_starts):
        if time.time() > deadline - 5:
            break
        
        # Random start
        x = rng.normal(0, 0.1, N)
        x -= np.mean(x)  # Ensure sum is 0
        x = np.clip(x, -0.45, 0.45)
        
        result = minimize(
            _objective,
            x,
            jac=_objective_grad,
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * N,
            options={'maxiter': 200, 'ftol': 1e-10}
        )
        
        h = 0.5 + result.x
        h = np.clip(h, 0, 1)
        
        # Renormalize to ensure sum is N/2
        if np.sum(h) > 0:
            h = h * (N/2) / np.sum(h)
            h = np.clip(h, 0, 1)
        
        val = _max_overlap(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    return best_h
