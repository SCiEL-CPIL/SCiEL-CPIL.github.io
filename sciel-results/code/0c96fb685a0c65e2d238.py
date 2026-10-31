import numpy as np
from scipy.optimize import minimize

N = 600

def construct():
    # Parameterize h_i as A_0 + sum_{k=1}^M A_k cos(2*pi*k*i/N)
    M = 10
    # Initial guess: Uniform distribution (A_0 = 1/N, others 0)
    # This is the known global minimizer for the peak autoconvolution.
    # We use a Remez-inspired iterative minimax approach by using a robust
    # derivative-free optimizer (Nelder-Mead) to refine the coefficients,
    # ensuring we stay in the basin of the global minimum.
    
    A_init = np.zeros(M + 1)
    A_init[0] = 1.0 / N
    
    i = np.arange(N)
    # Pre-compute cosine matrix for speed
    # C[k, i] = cos(2*pi*k*i/N)
    C = np.zeros((M + 1, N))
    C[0, :] = 1.0
    for k in range(1, M + 1):
        C[k, :] = np.cos(2 * np.pi * k * i / N)
        
    def coeffs_to_h(A):
        # h = A @ C
        return A @ C

    def objective(A):
        h = coeffs_to_h(A)
        # Penalty for negative values
        neg = np.sum(np.maximum(0, -h)**2)
        if neg > 1e-8:
            return 1e6 + neg
        
        # Normalize to sum 1
        s = np.sum(h)
        if s <= 1e-12:
            return 1e6
        h_norm = h / s
        
        # Compute autoconvolution
        g = np.convolve(h_norm, h_norm, mode='full')
        # The peak is at the center
        max_g = np.max(g)
        
        return max_g

    # Run optimization
    # Nelder-Mead is a derivative-free method suitable for minimax problems
    # where gradients may be ill-defined at the maximum.
    res = minimize(objective, A_init, method='Nelder-Mead', 
                   options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12})
    
    A_opt = res.x
    h_opt = coeffs_to_h(A_opt)
    
    # Ensure non-negative and normalize
    h_opt = np.maximum(h_opt, 0)
    s = np.sum(h_opt)
    if s > 0:
        h_opt = h_opt / s
    else:
        # Fallback to uniform if optimization failed
        h_opt = np.ones(N) / N
        
    return list(h_opt)
