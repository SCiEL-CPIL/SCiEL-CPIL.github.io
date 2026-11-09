import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression for a nonlinear oscillator using a Rational Function (Padé-like) basis.
    
    Strategy:
    The previous approach used an additive combination of polynomial and tanh terms.
    This approach uses a rational function form:
    y = N(x, v) / D(x, v)
    
    Where N is a polynomial in x and v up to degree 4 (using 5 parameters p0-p4)
    and D is a constant plus a polynomial in x and v (using parameters p5-p9).
    
    This structure is often more stable for BFGS because:
    1. The denominator prevents the output from growing too large (saturation effect).
    2. It can model non-linear interactions more efficiently than high-degree polynomials.
    3. It is a closed-form, smooth function suitable for gradient-based optimization.
    
    Inputs:
    x: np.ndarray (position)
    v: np.ndarray (velocity)
    params: np.ndarray of length 10
    
    Output:
    np.ndarray of predictions
    """
    p = params
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = p
    
    # Pre-compute powers of x and v to speed up evaluation
    # We limit the degree to keep the number of terms manageable and avoid overfitting/ill-conditioning
    # Numerator: Linear combination of monomials x^i * v^j
    # Let's use a basis that covers the main interactions:
    # 1: 1
    # 2: x
    # 3: v
    # 4: x*v
    # 5: x^2 + v^2 (or similar quadratic form) -- actually, let's stick to specific monomials
    # To use exactly 10 params, let's define:
    # N = p0 + p1*x + p2*v + p3*x*v + p4*x^2
    # D = 1 + p5*x^2 + p6*v^2 + p7*x*v + p8*x^3 + p9*v^3
    
    # This uses all 10 parameters.
    # The denominator ensures that as x or v grow, the fraction is bounded,
    # which is physically reasonable for oscillator dynamics (bounded energy/state).
    
    x2 = x * x
    v2 = v * v
    x3 = x2 * x
    v3 = v2 * v
    xv = x * v
    
    # Numerator
    num = p0 + p1 * x + p2 * v + p3 * xv + p4 * x2
    
    # Denominator
    # Start with 1.0 to avoid division by zero initially and ensure stability
    # If the fit drives the denominator to 0, the loss becomes infinite, which BFGS should avoid.
    den = 1.0 + p5 * x2 + p6 * v2 + p7 * xv + p8 * x3 + p9 * v3
    
    # Safety check: if denominator is too close to zero, replace with a small epsilon
    # to prevent numerical instability (division by zero or near-zero).
    # However, standard practice in these tasks is to let the optimizer handle it via loss,
    # but for robustness during the 120s window, we clip the denominator.
    # We use a small epsilon to prevent inf/nan.
    eps = 1e-8
    den_safe = np.where(np.abs(den) < eps, eps * np.sign(den), den)
    # If den is exactly 0, sign is 0, so we handle that case:
    den_safe = np.where(den == 0, eps, den_safe)
    
    result = num / den_safe
    
    return result
