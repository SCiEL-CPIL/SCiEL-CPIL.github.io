"""Direct construction using the Cohn-Goncalves Laguerre polynomial root structure."""
import numpy as np
import time
from numpy.polynomial.laguerre import laggauss
from scipy.special import genlaguerre, eval_genlaguerre
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = 600.0
_SAFETY_MARGIN = 30.0

def _deadline_reached():
    return (time.time() - _START_TIME) >= (_DEADLINE - _SAFETY_MARGIN)

def _compute_c4_bound(roots, m=12, n=1):
    """
    Compute the upper bound C4 for the given double-root locations.
    
    The Cohn-Goncalves construction with m double roots {x_i} and n=1
    defines a polynomial whose sign changes determine the uncertainty bound.
    C4 is related to the integral of the squared Laguerre-weighted polynomial.
    
    For the refined construction, C4 involves:
    - The product P(x) = prod_i (x - x_i)^2
    - Weighted by the Laguerre weight e^{-x} x^{2n-1} = e^{-x} x
    - The bound involves the L2 norm and sign change structure
    """
    m = 12
    n = 1
    alpha = 2 * n - 1  # = 1
    
    # Evaluate the polynomial P(x) = prod (x - x_i)^2 at quadrature points
    # Use Gauss-Laguerre quadrature for the integral
    # We need enough points to resolve the polynomial of degree 2m = 24
    n_quad = 60  # sufficient for degree 2m with weight x^alpha e^{-x}
    
    # Get quadrature nodes and weights for weight x^alpha * e^{-x}
    # Standard Gauss-Laguerre is for alpha=0; for general alpha use generalized
    # Use numpy's laggauss for alpha=0 and adjust, or use scipy
    # For simplicity and robustness, use a fine grid with trapezoidal rule
    # in a transformed variable, or use Gauss-Laguerre with correction
    
    # Better: use the fact that for generalized Laguerre weight x^alpha e^{-x},
    # we can use Gauss-Laguerre nodes for alpha=0 and multiply by x^alpha
    x_nodes, w_nodes = laggauss(n_quad)
    # Adjust weights for the generalized weight x^alpha e^{-x}
    w_adj = w_nodes * x_nodes**alpha
    
    # Evaluate P(x) = prod_i (x - x_i)^2 at nodes
    # P(x) is degree 2m = 24
    P_vals = np.ones(n_quad)
    for xi in roots:
        P_vals *= (x_nodes - xi)**2
    
    # The Laguerre-weighted L2 norm squared:
    # I = integral_0^inf P(x) * x^alpha * e^{-x} dx
    # Approximate via quadrature
    I = np.sum(P_vals * w_adj)
    
    # The uncertainty bound C4 for the Fourier sign problem with m sign changes
    # involves the ratio of the L2 norm to the product structure
    # For the refined Cohn-Goncalves, C4 ~ sqrt(I) / (some normalization)
    # The key quantity to minimize is the weighted norm
    # A tighter bound comes from smaller I
    
    # Additionally, the last sign change location matters
    # The bound C4 is proportional to sqrt(I) times a factor depending on
    # the rightmost root location
    
    # For a more accurate C4 estimate, we also need to account for
    # the sign structure. The polynomial P(x) = prod (x-xi)^2 is always >= 0,
    # so the sign changes come from the Laguerre polynomial factor.
    # The refined bound involves the interplay between P(x) and L_m(x)
    
    # A practical surrogate that correlates with C4:
    # C4_proxy = sqrt(I) / (prod of root spacings factor)
    # But the simplest effective objective is just I (or sqrt(I))
    
    # Return the bound (lower is better)
    return np.sqrt(I)

def construct():
    """
    Construct the 12 positive double-root locations for the Cohn-Goncalves
    Laguerre polynomial construction with m=12, n=1.
    
    Optimizes the root positions to minimize the C4 uncertainty upper bound.
    """
    m = 12
    n = 1
    
    # Initial guess: zeros of L_{2m}^{(2n-1)} = L_{24}^{(1)}
    degree = 2 * m
    alpha = 2 * n - 1  # = 1
    
    try:
        L = genlaguerre(degree, alpha)
        roots_init = L.roots()
        roots_init = roots_init[np.real(roots_init) > 0]
        roots_init = np.real(roots_init)
        roots_init = np.sort(roots_init)
        
        if len(roots_init) != m:
            if len(roots_init) > m:
                roots_init = roots_init[:m]
            else:
                roots_init = _fallback_construct(m, n)
    except Exception:
        roots_init = _fallback_construct(m, n)
    
    # Ensure we have exactly m positive roots
    roots_init = np.asarray(roots_init, dtype=float)
    if len(roots_init) < m:
        # Pad with reasonable values
        extra = np.linspace(roots_init[-1] * 1.5, roots_init[-1] * 2.0, m - len(roots_init))
        roots_init = np.concatenate([roots_init, extra])
    elif len(roots_init) > m:
        roots_init = roots_init[:m]
    
    # Ensure sorted and positive
    roots_init = np.sort(roots_init)
    roots_init = np.maximum(roots_init, 1e-6)
    
    # Optimize to minimize C4 bound
    best_roots = roots_init.copy()
    best_val = _compute_c4_bound(best_roots, m, n)
    
    # Use L-BFGS-B for optimization with positive constraints
    try:
        def objective(x):
            if _deadline_reached():
                return best_val
            # Ensure positive and sorted by using soft constraints
            # Use log-space to enforce positivity
            x_pos = np.exp(x)  # x in log-space
            # Sort to maintain ordering
            x_sorted = np.sort(x_pos)
            return _compute_c4_bound(x_sorted, m, n)
        
        # Initial guess in log-space
        x0 = np.log(roots_init)
        
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if result.success or result.fun < best_val:
            x_opt = np.exp(result.x)
            x_opt = np.sort(x_opt)
            x_opt = np.maximum(x_opt, 1e-6)
            val_opt = _compute_c4_bound(x_opt, m, n)
            if val_opt < best_val:
                best_roots = x_opt
                best_val = val_opt
        
    except Exception:
        pass
    
    # Try a second optimization pass with different initial perturbation if time allows
    if not _deadline_reached():
        try:
            # Perturb the best solution
            np.random.seed(42)
            x_perturbed = best_roots * (1 + 0.1 * np.random.randn(m))
            x_perturbed = np.sort(x_perturbed)
            x_perturbed = np.maximum(x_perturbed, 1e-6)
            
            val_perturbed = _compute_c4_bound(x_perturbed, m, n)
            
            if val_perturbed < best_val:
                best_roots = x_perturbed.copy()
                best_val = val_perturbed
            
            # Optimize from perturbed starting point
            def objective2(x):
                if _deadline_reached():
                    return best_val
                x_pos = np.exp(x)
                x_sorted = np.sort(x_pos)
                return _compute_c4_bound(x_sorted, m, n)
            
            x0_2 = np.log(best_roots)
            result2 = minimize(objective2, x0_2, method='L-BFGS-B',
                             options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if result2.success or result2.fun < best_val:
                x_opt2 = np.exp(result2.x)
                x_opt2 = np.sort(x_opt2)
                x_opt2 = np.maximum(x_opt2, 1e-6)
                val_opt2 = _compute_c4_bound(x_opt2, m, n)
                if val_opt2 < best_val:
                    best_roots = x_opt2
                    best_val = val_opt2
                    
        except Exception:
            pass
    
    # Final verification
    best_roots = np.sort(best_roots)
    best_roots = np.maximum(best_roots, 1e-10)
    
    return [float(x) for x in best_roots]

def _fallback_construct(m, n):
    """Fallback: use Laguerre-Gauss quadrature nodes as proxy."""
    try:
        x, w = laggauss(m, alpha=2*n-1)
        return np.sort(x)
    except Exception:
        return np.array([4.0 * (i + 1) for i in range(m)], dtype=float)

# EVOLVE-BLOCK-END
