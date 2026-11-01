"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _c3_upper_bound(h):
    """
    Compute the upper bound on C3 for a signed step function with heights h.
    Metric: max_k |conv(h, h)[k]| / (sum(|h|))^2
    """
    # We will minimize this value using L-BFGS-B on a smooth surrogate.
    # We use FFT for fast convolution.
    
    def objective(x):
        h = np.tanh(x)
        # Use FFT for convolution
        # Length of conv is 2N-1. We need at least 2N-1 points for FFT to avoid circular convolution artifacts.
        # Next power of 2 for efficiency.
        n_fft = 2**int(np.ceil(np.log2(2*N-1)))
        H = np.fft.rfft(h, n=n_fft)
        C = np.fft.irfft(H * H, n=n_fft)
        # The linear convolution is in C[0:2N-1]
        conv = C[:2*N-1]
        
        # Smooth max
        eps = 1e-8
        abs_conv = np.sqrt(conv**2 + eps)
        T = 5.0
        max_abs = np.max(abs_conv)
        log_sum_exp = np.log(np.sum(np.exp((abs_conv - max_abs) / T))) + max_abs / T
        log_max = T * log_sum_exp
        
        sum_abs_h = np.sum(np.sqrt(h**2 + eps))
        value = log_max / (sum_abs_h**2)
        return value

    # Initial guess: all ones
    x0 = np.full(N, 5.0)
    
    # Deadline guard
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin
    
    # Run optimization
    try:
        res = minimize(objective, x0, method='L-BFGS-B', 
                       options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False})
        h_opt = np.tanh(res.x)
    except Exception:
        h_opt = np.ones(N)
        
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return np.ones(N)
        
    # Optional: Discrete refinement (optional, but let's stick to continuous for now as per directive)
    # If we have time, we could try a few random restarts or local polish.
    # For now, return the continuous solution.
    
    return h_opt


def construct():
    return _c3_upper_bound(np.full(N, 5.0))
# EVOLVE-BLOCK-END
