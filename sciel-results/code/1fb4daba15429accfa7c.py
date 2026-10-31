import numpy as np
from scipy.optimize import minimize

N = 600

def construct():
    # Parameterize h_i as A_0 + sum_{k=1}^M A_k cos(2*pi*k*i/N)
    # Increase M to allow for sharper features (like a triangle)
    M = 20
    
    i = np.arange(N)
    # Pre-compute cosine matrix for speed
    C = np.zeros((M + 1, N))
    C[0, :] = 1.0
    for k in range(1, M + 1):
        C[k, :] = np.cos(2 * np.pi * k * i / N)
        
    def coeffs_to_h(A):
        # h = A @ C
        return A @ C

    def objective(A):
        h = coeffs_to_h(A)
        # Penalty for negative values (strict)
        neg = np.sum(np.maximum(0, -h)**2)
        if neg > 1e-6:
            return 1e6 + neg
        
        # Normalize to max(h) = 1
        # This makes the objective scale-invariant and matches the 
        # "normalized maximum" interpretation where we minimize 
        # max(conv) / max(h)
        max_h = np.max(h)
        if max_h <= 1e-12:
            return 1e6
        h_norm = h / max_h
        
        # Compute autoconvolution
        g = np.convolve(h_norm, h_norm, mode='full')
        max_g = np.max(g)
        
        return max_g

    # Initial guess: A triangle-like shape.
    # A triangle has Fourier coefficients ~ 1/k^2.
    # We approximate this with the first few harmonics.
    A_init = np.zeros(M + 1)
    # A rough triangle approximation:
    # f(x) = 1 - |x| on [-1,1] has coeff c_k ~ (-1)^k / (k^2-1)
    # We want a bump centered at 0.
    # Let's just start with a small cosine bump to break symmetry
    # and let the optimizer find the shape.
    # Or, initialize with a known good shape: "Gaussian-like"
    # A_0 = 0.5, A_1 = 0.4, A_2 = 0.1 ...
    
    # Better: Start with Uniform (A_0=1) and add a small perturbation
    # to move away from the local maximum (for the Ratio metric).
    A_init[0] = 1.0
    A_init[1] = 0.1
    A_init[2] = 0.05
    
    # Run optimization
    # Nelder-Mead is robust for this kind of non-smooth (max) objective.
    res = minimize(objective, A_init, method='Nelder-Mead', 
                   options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    A_opt = res.x
    h_opt = coeffs_to_h(A_opt)
    
    # Ensure non-negative
    h_opt = np.maximum(h_opt, 0)
    
    # Normalize to max 1 for final output consistency (though score is ratio)
    max_h = np.max(h_opt)
    if max_h > 0:
        h_opt = h_opt / max_h
    else:
        # Fallback to uniform
        h_opt = np.ones(N) / N
        
    return list(h_opt)
