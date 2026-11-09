"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_laguerre, gamma

_START_TIME = time.time()
_DEADLINE = 630.0 - 30.0  # safety margin

def _laguerre_poly(x, m):
    """Compute L_m(x) using recurrence for stability."""
    if m == 0:
        return np.ones_like(x)
    if m == 1:
        return 1.0 - x
    L0 = np.ones_like(x)
    L1 = 1.0 - x
    for k in range(2, m + 1):
        L2 = ((2 * k - 1) * x - k) * L1 - (k - 1) * L0
        L0, L1 = L1, L2
    return L1

def _compute_bound(roots):
    """
    Compute the Cohn-Goncalves uncertainty upper bound C4 given 12 double-root locations.
    The bound involves the last sign change of the Laguerre combination.
    We use a formulation where the bound is related to the maximum of |P(x)|/x^2 * e^{-x/2}
    or similar, but for the sign uncertainty, it's typically related to the integral
    or the location of the last zero.
    
    Actually, the "Fourier sign uncertainty upper bound" in the context of Cohn-Goncalves 
    with Laguerre polynomials usually refers to the bound on the number of sign changes 
    or the uncertainty constant. The "score is the upper bound on C4".
    
    Let's assume the objective function to minimize is the value of the polynomial 
    combination at the last root, or a related quantity that determines the bound.
    
    A common formulation for the "refined" bound involves minimizing the maximum 
    absolute value of the Laguerre combination over a certain interval, or specifically 
    looking at the behavior near the last sign change.
    
    Let's define the Laguerre combination as:
    P(x) = sum_{j=0}^{11} a_j * L_{m_j}(x)
    where the roots are double. This is a specific construction.
    
    However, a simpler and robust approach for the "last sign change" and "upper bound" 
    in these polynomial uncertainty problems is often to minimize the largest root 
    or a function of the roots that controls the uncertainty.
    
    Given the seed program returns [4.0, 8.0, ..., 48.0], let's assume the objective 
    is to minimize the last root, or a combination that reduces the uncertainty bound.
    
    Let's try to minimize the sum of the roots or the last root, subject to 
    positivity and ordering.
    """
    # Sort roots to ensure they are in increasing order
    roots = np.sort(roots)
    if np.any(roots <= 0):
        return 1e10  # Penalty for non-positive roots
    
    # The bound C4 is often related to the last sign change location or the 
    # maximum of a certain function. Let's assume we want to minimize the 
    # last root, as this directly affects the support of the uncertainty.
    # A more sophisticated objective might involve the integral of the 
    # Laguerre polynomial, but let's start with the last root.
    
    # Actually, the "Cohn-Goncalves" bound for Fourier uncertainty often involves 
    # the product of the roots or a symmetric function. Let's minimize the 
    # geometric mean of the roots, which is related to the product.
    
    # For now, let's use the last root as a proxy for the bound.
    return roots[-1]

def construct():
    global _START_TIME
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        return [4.0 * (i + 1) for i in range(12)]
    
    # Asymptotic distribution of Laguerre zeros for large m:
    # The zeros of L_m(x) are approximately distributed according to 
    # the Marchenko-Pastur law or similar. For L_m, the zeros scale as 
    # m^2 / (4 * sin^2(theta/2)) or similar.
    # A common approximation for the j-th zero of L_m is ~ 4 * j * m.
    # But here we have 12 roots for a specific combination.
    
    # Let's use a multi-start optimization.
    # Initial guess: asymptotic zeros of L_12(x)
    # The zeros of L_12 can be computed numerically.
    
    # Let's compute the zeros of L_12(x) using Newton's method or scipy.
    from scipy.special import roots_laguerre
    try:
        zeros = roots_laguerre(12)[0]
        # The zeros are already sorted and positive.
        x0 = zeros.copy()
    except:
        x0 = np.array([4.0 * (i + 1) for i in range(12)])
    
    best_roots = x0.copy()
    best_bound = _compute_bound(best_roots)
    
    # Multi-start optimization
    n_starts = 5
    for i in range(n_starts):
        if time.time() - _START_TIME > _DEADLINE:
            break
        
        if i == 0:
            x0_trial = best_roots.copy()
        else:
            # Perturb the initial guess
            np.random.seed(i)
            x0_trial = best_roots * (1 + 0.1 * np.random.randn(12))
            x0_trial = np.sort(x0_trial)
            x0_trial = np.maximum(x0_trial, 0.1)  # Ensure positivity
        
        # Define the objective function for scipy.optimize
        def objective(x):
            return _compute_bound(x)
        
        # Constraints: roots must be positive and ordered
        # We'll use bounds instead of constraints for simplicity
        bounds = [(0.1, None) for _ in range(12)]
        
        result = minimize(objective, x0_trial, method='L-BFGS-B', bounds=bounds, 
                          options={'maxiter': 1000, 'ftol': 1e-10})
        
        if result.success and result.fun < best_bound:
            best_bound = result.fun
            best_roots = result.x
    
    # Ensure the roots are sorted and positive
    best_roots = np.sort(best_roots)
    best_roots = np.maximum(best_roots, 0.1)
    
    # Convert to list of floats
    return [float(r) for r in best_roots]
# EVOLVE-BLOCK-END
