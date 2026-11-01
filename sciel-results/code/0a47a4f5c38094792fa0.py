"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _c3_upper_bound(h):
    """
    Optimize the signed step function h to minimize the upper bound on C3.
    Metric: max_k |(h*h)_k| / (sum_i |h_i|)^2
    Uses FFT for efficient convolution and L-BFGS-B for optimization.
    """
    
    def objective(x):
        # Map x to h in (-1, 1)
        h = np.tanh(x)
        
        # Compute autoconvolution using FFT
        # Linear convolution of two length-N arrays requires n_fft >= 2N-1
        n_fft = 2 * N
        H = np.fft.rfft(h, n=n_fft)
        C = np.fft.irfft(H * H, n=n_fft)
        
        # C contains the linear convolution in the first 2N-1 samples
        # The rest are zero (or near zero due to numerical errors)
        # We consider all 2N samples for the max, though the last one is 0
        abs_C = np.sqrt(C**2 + 1e-8)
        
        # Soft max approximation of max(abs_C)
        # We want to minimize max(abs_C) / (sum(|h|))^2
        # Equivalent to minimizing log(max(abs_C)) - 2*log(sum(|h|))
        # Using soft-max for differentiability
        
        T = 5.0  # Temperature for soft max
        # Subtract max for numerical stability in exp
        max_abs_C = np.max(abs_C)
        log_sum_exp = np.log(np.sum(np.exp((abs_C - max_abs_C) / T))) + max_abs_C / T
        soft_max = T * log_sum_exp
        
        # Denominator: (sum |h|)^2
        abs_h = np.sqrt(h**2 + 1e-8)
        sum_abs_h = np.sum(abs_h)
        
        # Objective to minimize
        # If sum_abs_h is very small, the value blows up, which is good (penalty).
        val = soft_max / (sum_abs_h**2 + 1e-8)
        return val

    # Initial guess: all ones
    # tanh(x) = 1 => x is large positive.
    # Start with x = 5.0
    x0 = np.full(N, 5.0)
    
    # Use L-BFGS-B
    # FFT is fast, so we can allow more iterations
    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-6})
    
    h_opt = np.tanh(res.x)
    return h_opt


def construct():
    return _c3_upper_bound(np.full(N, 5.0))
# EVOLVE-BLOCK-END
