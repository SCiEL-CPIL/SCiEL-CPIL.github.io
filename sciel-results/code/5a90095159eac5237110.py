import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Global rational function model using Chebyshev polynomials.
    
    f(x) = N(x) / D(x)
    where N(x) = sum_{i=0}^{4} a_i T_i(x) and D(x) = sum_{j=0}^{4} b_j T_j(x)
    
    This allows the model to capture singularities, poles, and asymptotic
    behaviors that polynomial or radial-basis models cannot represent well.
    The Chebyshev basis provides numerical stability on [-1, 1].
    We map x in [-2, 2] to t in [-1, 1] via t = x/2.
    
    Parameters (10 total):
    params[0..4]  = a0..a4  (numerator coefficients)
    params[5..9]  = b0..b4  (denominator coefficients)
    
    To avoid division by zero, we add a small epsilon to the denominator.
    The initial value for all params is 1.0, giving N(x)=sum T_i, D(x)=sum T_i,
    so f(x)=1 initially, which is a safe starting point.
    """
    a = params[:5]
    b = params[5:]
    
    # Map x from [-2, 2] to t in [-1, 1]
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 through T_4
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2*t^2 - 1
    # T_3(t) = 4*t^3 - 3*t
    # T_4(t) = 8*t^4 - 8*t^2 + 1
    t2 = t * t
    t3 = t2 * t
    t4 = t2 * t2
    
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t2 - 1.0
    T3 = 4.0 * t3 - 3.0 * t
    T4 = 8.0 * t4 - 8.0 * t2 + 1.0
    
    # Build numerator and denominator
    num = a[0] * T0 + a[1] * T1 + a[2] * T2 + a[3] * T3 + a[4] * T4
    den = b[0] * T0 + b[1] * T1 + b[2] * T2 + b[3] * T3 + b[4] * T4
    
    # Add a small regularization to avoid division by zero
    # The sign of epsilon should match the sign of den to avoid flipping
    # A robust approach: use np.where to handle near-zero denominators
    eps = 1e-8
    den_safe = np.where(np.abs(den) < eps, np.sign(den) * eps + (den == 0) * eps, den)
    
    result = num / den_safe
    
    # Replace any non-finite values with a large penalty value
    # This helps the optimizer avoid bad regions
    result = np.where(np.isfinite(result), result, 1e10)
    
    return result
