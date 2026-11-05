import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.special import eval_chebyt

N = 400

def construct():
    deadline = time.time() + 115  # Safety margin for 120s limit

    # Precompute Chebyshev basis
    # Map index i in [0, N-1] to x in [-1, 1]
    x = np.linspace(-1, 1, N)
    
    # We use degrees 0 to K. 
    # Degree 0 is constant (sum control).
    # Higher degrees allow shaping.
    # Let's try up to degree 10 or 12.
    K = 12
    basis = np.zeros((N, K + 1))
    for k in range(K + 1):
        basis[:, k] = eval_chebyt(k, x)

    def compute_value(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return float('inf')
        if not np.isfinite(s):
            return float('inf')
        z = np.convolve(h, h)
        return abs(2 * N * np.max(z) / (s * s))

    best_h = np.ones(N)
    best_val = compute_value(best_h) # 2.0

    def objective(coeffs):
        if time.time() > deadline - 5:
            return float('inf')
        h = coeffs @ basis.T
        # Check for NaN/Inf
        if not np.all(np.isfinite(h)):
            return float('inf')
        return compute_value(h)

    # Bounds for coefficients
    # We need to ensure sum(h) != 0.
    # sum(h) = coeffs[0] * sum(basis[:, 0]) + ...
    # sum(basis[:, 0]) = N.
    # sum(basis[:, k]) for k>=1 is small (oscillatory).
    # So coeffs[0] roughly controls the sum.
    # If coeffs[0] is close to 0, sum might be small or zero.
    # We set bounds to encourage non-zero sum but allow shaping.
    
    # Let's set bounds for c0 (constant) to be positive to ensure sum > 0 roughly.
    # Actually, if c0 is negative, sum is negative, s^2 is same.
    # But if c0 is too small, higher terms might dominate and make sum zero.
    # Let's constrain c0 to [0.1, 2.0] and others to [-1.0, 1.0].
    # This keeps the "mean" positive and significant.
    
    bounds = []
    bounds.append((0.1, 2.0)) # c0
    for k in range(1, K + 1):
        bounds.append((-1.0, 1.0))
        
    # Differential Evolution is global but slow.
    # popsize=15, maxiter=500 might be too slow for 13 dims?
    # Let's try a smaller popsize and maxiter, relying on the landscape being smooth enough.
    # Or reduce K.
    
    # Let's try K=8 first to be safe on time.
    K_opt = 8
    bounds_opt = [(0.1, 2.0)] + [(-1.0, 1.0)] * K_opt
    basis_opt = basis[:, :K_opt + 1]
    
    def objective_opt(coeffs):
        if time.time() > deadline - 5:
            return float('inf')
        h = coeffs @ basis_opt.T
        if not np.all(np.isfinite(h)):
            return float('inf')
        return compute_value(h)

    try:
        # tol=0.1 is loose to finish fast.
        # seed=42 for reproducibility
        res = differential_evolution(
            objective_opt, 
            bounds_opt, 
            seed=42,
            maxiter=100,
            popsize=10,
            tol=0.01,
            polish=True # Local refinement at the end
        )
        
        if res.fun < best_val:
            best_val = res.fun
            best_h = res.x @ basis_opt.T
            
    except Exception:
        pass

    # If we have time left, try a slightly larger K or different bounds?
    # Or just return what we have.
    # Given the "STAGNATION" warning, a structural change is key.
    # The Chebyshev basis is structurally different from Gaussian bumps.
    
    return best_h
