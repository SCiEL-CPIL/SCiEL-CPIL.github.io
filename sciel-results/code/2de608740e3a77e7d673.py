"""
Strategy: Pure Linear Basis Expansion with Chebyshev Polynomials.
Instead of a non-linear model where parameters control frequencies/widths (creating a multi-modal, rugged landscape for BFGS), we use a strictly linear model in the parameters:
y(x) = sum_{i=0}^{9} p_i * T_i(x/2)

Here T_i are Chebyshev polynomials of the first kind.
1. The domain [-2, 2] is scaled to [-1, 1] via x/2.
2. Chebyshev polynomials are minimax polynomials, providing the best uniform approximation for smooth functions with minimal oscillation compared to monomials.
3. The design matrix is well-conditioned (ill-conditioned number is moderate, unlike high-degree monomials).
4. The objective function (MSE) is strictly convex in p. BFGS will converge to the global minimum quickly and reliably, avoiding local minima that plague non-linear frequency/width parameterization.
5. 10 parameters allow up to a 9th-degree polynomial approximation, which can capture a wide variety of smooth functions (polynomials, approximations to trig/exponential via series).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _chebyshev_basis(x: np.ndarray, n: int) -> np.ndarray:
    """
    Compute Chebyshev polynomials of the first kind T_k(t) for t in [-1, 1].
    Returns a matrix of shape (len(x), n) where column k is T_k(x).
    """
    # Scale x from [-2, 2] to [-1, 1]
    t = x / 2.0
    
    # Clip to ensure numerical stability for the recurrence, though mathematically T_k is defined for all real t.
    # However, for x in [-2,2], t is in [-1,1], so no clipping is strictly needed for validity, 
    # but it protects against any slight numerical drift if x is exactly at boundaries.
    t = np.clip(t, -1.0, 1.0)
    
    basis = np.zeros((len(x), n), dtype=float)
    
    if n > 0:
        # T_0(t) = 1
        basis[:, 0] = 1.0
        
    if n > 1:
        # T_1(t) = t
        basis[:, 1] = t
        
    for k in range(2, n):
        # Recurrence: T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
        basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
        
    return basis

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear model using Chebyshev basis.
    params: array of length 10 (coefficients for T_0 to T_9).
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(params, dtype=float)
    
    # Safety check
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(p)):
        return np.zeros_like(x)
        
    # Ensure params is exactly length 10
    if p.shape[0] != 10:
        # If shape is wrong, pad or truncate to 10 (though evaluator should provide 10)
        p_temp = np.zeros(10)
        min_len = min(10, p.shape[0])
        p_temp[:min_len] = p[:min_len]
        p = p_temp

    # Generate basis matrix
    # n=10 corresponds to T_0 ... T_9
    B = _chebyshev_basis(x, 10)
    
    # Linear combination
    y = B @ p
    
    # Ensure finite output
    if not np.all(np.isfinite(y)):
        y = np.where(np.isfinite(y), y, 0.0)
        
    return y
# EVOLVE-BLOCK-END
