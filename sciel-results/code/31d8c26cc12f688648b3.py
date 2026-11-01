"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _c3_upper_bound(h):
    """
    Compute the upper bound on C3 for a signed step function with heights h.
    
    We assume the metric is the ratio of the L1 norm of the autoconvolution 
    to the square of the L1 norm of h:
    V(h) = sum_k |(h*h)_k| / (sum_i |h_i|)^2
    
    This metric is improved by using signs that cause cancellations in the convolution,
    reducing the sum of absolute values of the convolution terms.
    """
    def objective(x):
        h = np.tanh(x)  # Bounded in (-1, 1)
        conv = np.convolve(h, h, mode='full')
        # Smooth L1 norm of convolution
        eps = 1e-6
        sum_abs_conv = np.sum(np.sqrt(conv**2 + eps))
        # Smooth L1 norm of h
        sum_abs_h = np.sum(np.sqrt(h**2 + eps))
        value = sum_abs_conv / (sum_abs_h**2)
        return value
    
    # Initialize with random signs to escape the all-positive local minimum
    # which has high L1 norm of convolution due to lack of cancellation.
    np.random.seed(42)
    x0 = np.random.randn(N) * 2.0
    
    # Use L-BFGS-B
    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
    
    h_opt = np.tanh(res.x)
    return h_opt


def construct():
    return _c3_upper_bound(np.random.randn(N) * 2.0)
# EVOLVE-BLOCK-END
