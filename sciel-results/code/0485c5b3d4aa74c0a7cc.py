# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

N = 400

def _objective(x):
    """
    Objective function to minimize.
    x is the parameter vector.
    h = tanh(x) ensures h is in (-1, 1).
    Metric: max_k |conv(h, h)_k| / (sum |h_i|)^2
    """
    h = np.tanh(x)
    # Compute autoconvolution
    conv = np.convolve(h, h, mode='full')
    
    # Smooth absolute value
    eps = 1e-8
    abs_conv = np.sqrt(conv**2 + eps)
    
    # Smooth max using log-sum-exp
    # T is the temperature; lower T is closer to true max but less stable
    T = 1.0 
    max_abs = np.max(abs_conv)
    # Subtract max for numerical stability
    log_sum_exp = np.log(np.sum(np.exp((abs_conv - max_abs) / T))) + max_abs / T
    log_max = T * log_sum_exp
    
    # L1 norm of h
    sum_abs_h = np.sum(np.sqrt(h**2 + eps))
    
    # Avoid division by zero
    if sum_abs_h < 1e-6:
        return 1e10
        
    value = log_max / (sum_abs_h**2)
    return value

def _objective_grad(x):
    """
    Gradient of the objective function.
    This is complex to derive analytically with the smooth approximations.
    We can use finite differences if needed, but L-BFGS-B can estimate gradients.
    However, providing gradients speeds up convergence.
    For now, we rely on L-BFGS-B's internal gradient estimation or use 'BFGS' method which estimates.
    Actually, L-BFGS-B does not estimate gradients, it requires them or uses finite diff if not provided?
    No, scipy's minimize with L-BFGS-B requires a gradient if not provided, or it uses finite differences?
    Checking docs: L-BFGS-B uses finite difference approximation of the gradient if jac is not provided.
    This is expensive for N=400 (400 extra evaluations per step).
    Let's try to provide an analytical gradient or use a method that estimates it efficiently.
    Alternatively, we can use 'Powell' or 'Nelder-Mead' but they are slower.
    Let's stick to L-BFGS-B and let it estimate gradients, but limit iterations.
    Or, we can simplify the problem by fixing the scale.
    """
    pass # Not implementing gradient explicitly to keep code simple, relying on finite diff if necessary,
         # but finite diff for N=400 is 400x cost. 
         # Better to use a method that doesn't require gradients or implement a simple gradient.
         # Let's try to use 'L-BFGS-B' with 'jac' set to None (finite diff).
         # To speed it up, we can reduce the dimensionality or use a smarter initialization.

def construct():
    # Use multiple random restarts to find a better local minimum
    best_h = None
    best_val = float('inf')
    
    # Random seed for reproducibility in evaluation context if needed, but usually random is fine
    # We'll try a few random initializations
    for i in range(5):
        # Random initialization in the range [-5, 5] to cover tanh range well
        x0 = np.random.uniform(-5, 5, N)
        
        # Optimize
        res = minimize(_objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-8})
        
        h_opt = np.tanh(res.x)
        val = _objective(res.x)
        
        if val < best_val:
            best_val = val
            best_h = h_opt
            
    # If all optimizations fail or return NaN, fallback to seed
    if best_h is None or not np.all(np.isfinite(best_h)):
        return np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
