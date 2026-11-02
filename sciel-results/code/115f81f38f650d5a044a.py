import numpy as np
from scipy.optimize import minimize

N = 600

def construct():
    # We want to minimize the normalized maximum of the autoconvolution.
    # The normalized maximum is defined as max_t (h*h)(t) / (integral h)^2.
    # This quantity is scale-invariant. We can fix integral h = 1 and minimize max_t (h*h)(t).
    # Alternatively, we can fix sum(x_i) = 1 (in discrete units) and minimize max_k d_k,
    # where d_k is the discrete autoconvolution.
    # Let's fix sum(x) = 1.
    # x_i >= 0.
    # Objective: max_k (conv(x, x))[k].
    # The continuous autoconvolution max is approximately w * max_k d_k.
    # The integral is w * sum(x).
    # Ratio = (w * max d_k) / (w * sum x)^2 = max d_k / (w * (sum x)^2).
    # If we fix sum(x) = 1, ratio = max d_k / w.
    # Minimizing max d_k is equivalent to minimizing the ratio.
    
    def objective(x):
        # x is a vector of length N
        # Compute discrete autoconvolution
        d = np.convolve(x, x, mode='full')
        return np.max(d)
    
    # Constraint: sum(x) = 1
    def constraint_sum(x):
        return np.sum(x) - 1.0
    
    # Initial guess: uniform distribution
    x0 = np.ones(N) / N
    
    bounds = [(0, None)] * N
    
    cons = [{'type': 'eq', 'fun': constraint_sum}]
    
    # Use SLSQP
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                   options={'maxiter': 1000, 'ftol': 1e-10})
    
    x_opt = res.x
    
    # Ensure non-negativity and sum to 1
    x_opt = np.clip(x_opt, 0, None)
    x_opt /= np.sum(x_opt)
    
    return x_opt.tolist()
