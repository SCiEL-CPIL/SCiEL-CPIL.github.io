"""Hybrid rational-Chebyshev approximation for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational function with Chebyshev polynomial numerator and denominator.
    
    y = P(x) / (Q(x) + 1)
    
    P(x) = sum_{k=0}^{4} p_k * T_k(x/2)
    Q(x) = sum_{k=0}^{4} q_k * T_k(x/2)
    
    The +1 in the denominator ensures it is positive near x=0 when params start at 1,
    helping BFGS convergence. Chebyshev basis provides better conditioning than
    monomials for x in [-2, 2].
    
    params layout (length 10):
      params[0..4] = p_0, p_1, p_2, p_3, p_4 (numerator coefficients)
      params[5..9] = q_0, q_1, q_2, q_3, q_4 (denominator coefficients)
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    p = np.asarray(params, dtype=float)
    
    # Map x from [-2, 2] to t in [-1, 1]
    t = 0.5 * x
    
    # Compute Chebyshev polynomials T_0 to T_4
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    
    T0 = np.ones_like(t)
    T1 = t.copy()
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    
    # Numerator: P(x) = p_0*T_0 + p_1*T_1 + p_2*T_2 + p_3*T_3 + p_4*T_4
    P = p[0] * T0 + p[1] * T1 + p[2] * T2 + p[3] * T3 + p[4] * T4
    
    # Denominator: Q(x) + 1 where Q(x) = q_0*T_0 + q_1*T_1 + q_2*T_2 + q_3*T_3 + q_4*T_4
    Q = p[5] * T0 + p[6] * T1 + p[7] * T2 + p[8] * T3 + p[9] * T4
    D = Q + 1.0
    
    # Ensure denominator doesn't get too close to zero for numerical stability
    # Use a smooth soft floor that pushes D away from 0
    # When |D| is large, the correction is negligible; when |D| is small, it pushes away
    # den_safe = D + eps * exp(-|D| / scale) * sign(D + tiny)
    # But sign is not differentiable. Use a smoother approach:
    # D_safe = D + 0.01 * exp(-10 * D * D)
    # This is smooth (C-infinity) and pushes D away from 0 when D is near 0
    D_safe = D + 0.01 * np.exp(-10.0 * D * D)
    
    result = P / D_safe
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
