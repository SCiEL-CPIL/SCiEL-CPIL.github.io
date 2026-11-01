# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

N = 400

def _compute_value(h):
    """Compute the raw upper bound value for a given height vector h."""
    # h is expected to be in [-1, 1]
    conv = np.convolve(h, h, mode='full')
    abs_conv = np.abs(conv)
    max_conv = np.max(abs_conv)
    sum_abs_h = np.sum(np.abs(h))
    if sum_abs_h < 1e-6:
        return 1e6  # Avoid division by zero, return large penalty
    return max_conv / (sum_abs_h ** 2)

def _smoothed_objective(x, T=1.0, eps=1e-6):
    """
    Smoothed objective function for L-BFGS-B.
    x: raw parameters, h = tanh(x)
    """
    h = np.tanh(x)
    conv = np.convolve(h, h, mode='full')
    
    # Smooth absolute value
    abs_conv = np.sqrt(conv**2 + eps)
    abs_h = np.sqrt(h**2 + eps)
    
    # Softmax for max(abs_conv)
    # We want to approximate max(abs_conv)
    # logsumexp trick: T * log(sum(exp(abs_conv/T)))
    max_abs_conv = np.max(abs_conv)
    # Subtract max to prevent overflow
    log_sum_exp = np.log(np.sum(np.exp((abs_conv - max_abs_conv) / T))) + max_abs_conv / T
    soft_max = T * log_sum_exp
    
    sum_abs_h = np.sum(abs_h)
    denom = sum_abs_h ** 2
    
    if denom < 1e-6:
        return 1e6
        
    return soft_max / denom

def _gradient(x, T=1.0, eps=1e-6):
    """
    Numerical gradient or analytic? 
    L-BFGS-B can use finite differences if jac is not provided, 
    but analytic is faster. However, deriving the gradient of 
    softmax(conv(tanh(x))) is complex. 
    Given N=400, finite differences (2N+1 = 801 evaluations) per iteration 
    might be too slow if many iterations are needed. 
    But L-BFGS-B uses finite differences internally if jac=None. 
    Let's provide a numerical gradient using central differences for accuracy 
    or rely on scipy's default. 
    Actually, scipy's minimize with method='L-BFGS-B' and jac=None 
    uses forward differences. 
    Let's try to provide an analytic gradient if possible, or just let scipy handle it.
    For robustness and speed, let's try a simple numerical gradient 
    or just pass jac=None and hope it's fast enough. 
    With N=400, 400 function calls per gradient step. 
    If 100 iterations, 40000 calls. Each call is O(N^2) for conv? 
    np.convolve is O(N^2) or O(N log N)? 
    For N=400, O(N^2) is 160,000 ops. 
    40,000 * 160,000 = 6.4e9 ops. This might be too slow in Python/NumPy loop overhead? 
    np.convolve is C-optimized. 160k ops is very fast. 
    40k calls * 1ms/call = 40s. This is within 120s.
    """
    return None # Let scipy compute finite differences

def construct():
    # Initial guess: Random signs to escape local minima of "all ones"
    # Using random initialization is key for non-convex problems like this.
    # We want h to be close to +/- 1.
    # tanh(x) = +/- 1 when x = +/- large.
    # Let's initialize x with random values in [-5, 5] to get h in [-0.99, 0.99]
    np.random.seed(42) # Deterministic for reproducibility in evaluation? 
    # The task says "Choose a signed step function". 
    # If the evaluator runs construct() multiple times, we should be deterministic 
    # or return the best found. 
    # Since we have a time limit, we can run optimization once.
    
    x0 = np.random.uniform(-5, 5, N)
    
    # Define options for L-BFGS-B
    options = {
        'maxiter': 500,
        'ftol': 1e-12,
        'gtol': 1e-8,
        'maxls': 20
    }
    
    # Run optimization
    res = minimize(_smoothed_objective, x0, method='L-BFGS-B', jac=None, options=options)
    
    # Extract optimized h
    h_opt = np.tanh(res.x)
    
    # The task asks for a "signed step function". 
    # The continuous optimization might yield values like 0.8. 
    # Should we snap to +/- 1? 
    # The metric uses |h_i|. If we snap to 1, denominator increases, 
    # but numerator might change. 
    # Usually, for max-norm of convolution, extremal values are +/- 1.
    # Let's check if snapping improves the score.
    h_snapped = np.sign(h_opt)
    # Ensure no zeros become 1 arbitrarily? 
    # If h_opt is close to 0, sign is 0. 
    # But we want non-zero. 
    # Let's compare value of h_opt and h_snapped.
    
    val_opt = _compute_value(h_opt)
    val_snapped = _compute_value(h_snapped)
    
    if val_snapped < val_opt:
        return h_snapped
    else:
        return h_opt

# EVOLVE-BLOCK-END
