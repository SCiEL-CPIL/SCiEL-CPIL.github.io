"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_genlaguerre
from scipy.optimize import minimize_scalar
import time

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination
    (m=12, n=1) that minimize the C4 upper bound.
    
    Structural change: Instead of a fixed list of alpha values, use a
    continuous optimization over alpha (log-spaced) combined with local
    refinement. The product of roots of L_12^(alpha)(x) grows as alpha^12
    for large alpha, so we search over a wide range and refine.
    """
    start_time = time.time()
    deadline = 600.0
    safety_margin = 30.0
    
    m = 12
    
    def neg_log_product(alpha):
        """Negative log of product of roots (to minimize)."""
        try:
            x, w = roots_genlaguerre(m, alpha)
            if np.any(x <= 0):
                return 1e10
            return -np.sum(np.log(x))
        except Exception:
            return 1e10
    
    best_product = 0.0
    best_roots = None
    best_alpha = m
    
    # Coarse search over log-spaced alpha values
    alphas = np.logspace(0, 8, 200)  # from 1 to 1e8
    
    for alpha in alphas:
        if time.time() > deadline - safety_margin:
            break
        val = neg_log_product(alpha)
        if val < 1e10:
            product = np.exp(-val)
            if product > best_product:
                best_product = product
                best_alpha = alpha
                x, w = roots_genlaguerre(m, alpha)
                best_roots = [float(xi) for xi in x]
    
    # Local refinement around best alpha found
    if best_roots is not None:
        if time.time() < deadline - safety_margin:
            try:
                result = minimize_scalar(neg_log_product, 
                                         bounds=(best_alpha * 0.5, best_alpha * 2.0),
                                         method='bounded',
                                         options={'xatol': 1e-6, 'maxiter': 100})
                if result.success and result.fun < 1e10:
                    refined_alpha = result.x
                    x, w = roots_genlaguerre(m, refined_alpha)
                    if np.all(x > 0):
                        product = np.exp(np.sum(np.log(x)))
                        if product > best_product:
                            best_product = product
                            best_alpha = refined_alpha
                            best_roots = [float(xi) for xi in x]
            except Exception:
                pass
        
        # Additional refinement with tighter bounds
        if time.time() < deadline - safety_margin:
            try:
                result = minimize_scalar(neg_log_product,
                                         bounds=(best_alpha * 0.9, best_alpha * 1.1),
                                         method='bounded',
                                         options={'xatol': 1e-8, 'maxiter': 200})
                if result.success and result.fun < 1e10:
                    refined_alpha = result.x
                    x, w = roots_genlaguerre(m, refined_alpha)
                    if np.all(x > 0):
                        product = np.exp(np.sum(np.log(x)))
                        if product > best_product:
                            best_product = product
                            best_alpha = refined_alpha
                            best_roots = [float(xi) for xi in x]
            except Exception:
                pass
    
    # Fallback
    if best_roots is None:
        x, w = roots_genlaguerre(m, m)
        best_roots = [float(xi) for xi in x]
    
    best_roots.sort()
    return best_roots
# EVOLVE-BLOCK-END
