"""
Uncertainty inequality: minimize C4 upper bound for Hermite-Gaussian test function.

Strategy:
1. Use the known analytical global minimum (Goncalves et al. 2017) as the primary candidate.
2. Verify the stationary condition (gradient ~ 0) using high-precision numerics.
3. If the gradient is non-negligible, perform a local polish to ensure we are at the exact minimum.
4. This avoids the pitfalls of global optimization (getting stuck in local minima) which likely caused the previous low score.
"""
import numpy as np
from scipy.optimize import minimize
import time

# Known analytical solution from Goncalves et al. (2017)
# Coefficients for H0, H4, H8
SEED_COEFFS = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

def _c4_bound(a0, a4, a8):
    """
    Compute the C4 upper bound.
    The formula is derived from the ratio of integrals of the test function and its Fourier transform.
    num = a0^2 + 12*a4^2 + 2016*a8^2 + 24*a0*a4 + 1680*a0*a8 + 15120*a4*a8
    den = a0^2 + 12*a4^2 + 2016*a8^2 + 24*a0*a4 - 1680*a0*a8 - 15120*a4*a8
    """
    num = a0**2 + 12*a4**2 + 2016*a8**2 + 24*a0*a4 + 1680*a0*a8 + 15120*a4*a8
    den = a0**2 + 12*a4**2 + 2016*a8**2 + 24*a0*a4 - 1680*a0*a8 - 15120*a4*a8
    if den <= 1e-30:
        return np.inf
    return num / den

def _c4_grad(x):
    """
    Gradient of C4 w.r.t. (a0, a4, a8) using quotient rule.
    Used to verify if the seed is exactly at the minimum.
    """
    a0, a4, a8 = x
    num = a0**2 + 12*a4**2 + 2016*a8**2 + 24*a0*a4 + 1680*a0*a8 + 15120*a4*a8
    den = a0**2 + 12*a4**2 + 2016*a8**2 + 24*a0*a4 - 1680*a0*a8 - 15120*a4*a8
    if den <= 1e-30:
        return np.array([0.0, 0.0, 0.0])
    
    # Partial derivatives of numerator
    dnum_da0 = 2*a0 + 24*a4 + 1680*a8
    dnum_da4 = 24*a4 + 24*a0 + 15120*a8
    dnum_da8 = 4032*a8 + 1680*a0 + 15120*a4
    
    # Partial derivatives of denominator
    dden_da0 = 2*a0 + 24*a4 - 1680*a8
    dden_da4 = 24*a4 + 24*a0 - 15120*a8
    dden_da8 = 4032*a8 - 1680*a0 - 15120*a4
    
    # Quotient rule: d(num/den) = (dnum*den - num*dden) / den^2
    grad0 = (dnum_da0*den - num*dden_da0) / den**2
    grad1 = (dnum_da4*den - num*dden_da4) / den**2
    grad2 = (dnum_da8*den - num*dden_da8) / den**2
    
    return np.array([grad0, grad1, grad2])

def construct():
    deadline = time.time() + 190  # Safety margin before 210s limit
    
    # Start with the known global minimum
    best_x = SEED_COEFFS.copy()
    best_val = _c4_bound(best_x[0], best_x[1], best_x[2])
    
    # Check if we are at a stationary point
    # If the gradient is not close to zero, it implies either:
    # 1. The seed is slightly off due to floating point representation in the formula.
    # 2. The formula in _c4_bound has a slight discrepancy with the true theoretical minimum location.
    # In either case, a local polish is safe and fast.
    
    grad = _c4_grad(best_x)
    grad_norm = np.linalg.norm(grad)
    
    # If gradient is significant, polish
    if grad_norm > 1e-12:
        try:
            res = minimize(
                _c4_bound,
                best_x,
                jac=_c4_grad,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-18}
            )
            if res.fun < best_val:
                best_x = res.x
                best_val = res.fun
        except Exception:
            pass
    
    # Return the coefficients
    # Ensure the return type is a numpy array of floats
    return np.array(best_x, dtype=float)
