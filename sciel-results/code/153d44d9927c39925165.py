# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import sys

# Pre-computed RMS scales for basis functions on uniform [-3, 3]
# Index: 0:1, 1:x, 2:x^2, 3:x^3, 4:sin(x), 5:cos(x), 6:sin(2x), 7:cos(2x), 8:1/(1+x^2), 9:exp(-x^2/4)
_SCALES = np.array([
    1.0,                 # 1
    np.sqrt(3.0),        # x
    np.sqrt(81.0 / 5.0), # x^2
    np.sqrt(729.0 / 7.0),# x^3
    np.sqrt((6.0 - np.sin(6.0)) / 12.0), # sin(x)
    np.sqrt((6.0 + np.sin(6.0)) / 12.0), # cos(x)
    np.sqrt((12.0 - np.sin(12.0)) / 24.0), # sin(2x)
    np.sqrt((12.0 + np.sin(12.0)) / 24.0), # cos(2x)
    0.508,               # 1/(1+x^2) approx
    0.645                # exp(-x^2/4) approx
])

# Cache for basis matrix and target
_basis_cache = {"x": None, "V": None, "y": None}
_best_params = None
_start_time = None

def _get_basis(x):
    """Compute the basis matrix V for input x, using caching if possible."""
    global _basis_cache
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.ravel()
    
    # Check cache
    if _basis_cache["x"] is not None and np.array_equal(_basis_cache["x"], x):
        return _basis_cache["V"]
    
    n = x.size
    c0 = np.ones(n, dtype=np.float64)
    c1 = x
    c2 = x * x
    c3 = c2 * x
    s1 = np.sin(x)
    s2 = np.cos(x)
    s3 = np.sin(2.0 * x)
    s4 = np.cos(2.0 * x)
    s5 = 1.0 / (1.0 + c2)
    s6 = np.exp(-c2 / 4.0)
    
    V = np.empty((n, 10), dtype=np.float64)
    V[:, 0] = c0
    V[:, 1] = c1
    V[:, 2] = c2
    V[:, 3] = c3
    V[:, 4] = s1
    V[:, 5] = s2
    V[:, 6] = s3
    V[:, 7] = s4
    V[:, 8] = s5
    V[:, 9] = s6
    
    # Normalize columns by their pre-computed RMS scales
    V = V / _SCALES
    
    # Update cache
    _basis_cache["x"] = x
    _basis_cache["V"] = V
    
    return V

def _objective(params, V, y):
    """Objective function for optimization."""
    pred = V @ params
    return np.mean((pred - y) ** 2)

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate the hidden function using a diverse basis of 10 functions,
    normalized to have unit RMS on [-3, 3] to improve optimizer conditioning.
    Includes a final polish step with BFGS to refine parameters if time permits.
    """
    global _basis_cache, _best_params, _start_time
    
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.ravel()
    
    p = np.asarray(params, dtype=np.float64)
    if p.ndim > 1:
        p = p.ravel()
    
    # Ensure finite params
    p = np.where(np.isfinite(p), p, 1.0)
    p = np.clip(p, -1e6, 1e6)
    
    # Get basis matrix (cached)
    V = _get_basis(x)
    
    # If we don't have a target yet, this is likely the first call during fitting
    # We need to infer y. However, the interface only gives x and params.
    # The evaluator calls equation(x, params) to get predictions.
    # The evaluator fits params by minimizing MSE. 
    # Wait, the evaluator fits params. It calls equation(x_train, params) to get predictions,
    # compares to y_train (which it has), and computes MSE.
    # So equation(x, params) just needs to return V @ params.
    # The "polish" idea from the directive suggests we could do more, but we don't have y.
    # Unless... we are allowed to assume the evaluator calls this multiple times?
    # No, the evaluator calls equation(x, params) to get predictions.
    # The "polish" step in the directive usually applies when the function itself performs optimization.
    # Here, the *evaluator* performs the optimization.
    # So equation(x, params) is just a forward pass.
    # 
    # Re-reading the task: "The evaluator fits params... minimizing the training MSE."
    # "equation must return an array with one prediction per point."
    # So equation is just the model.
    # 
    # However, the "Strategy to try" says: "Implement a universal approximator... parameterized by the 10 available parameters via a fixed, deterministic initialization... to ensure the function remains closed-form and differentiable for BFGS optimization."
    # This implies we should change the *structure* of the model to be more expressive while still using the 10 params.
    # The current model is linear in params: V @ p.
    # A neural network with 10 params would be non-linear in params, which might be harder for BFGS but potentially more expressive.
    # But the directive says "KEEP this design and investigate improvements within its family."
    # The current design is a linear basis.
    # The inspirations use Legendre/Chebyshev + Trig.
    # The current program uses Poly + Trig + Rational + Gaussian.
    # 
    # Let's look at the "REFINEMENT" directive again:
    # "consider a smooth surrogate for a nonsmooth min/max objective... or an auxiliary variable t... or a change of variables... or a final high-precision polish pass"
    # 
    # If I cannot change the fact that the evaluator does the fitting, I can only change the mapping from params to predictions.
    # 
    # Let's try to combine the current basis with the Legendre/Chebyshev idea to ensure better conditioning or coverage.
    # The current basis is: 1, x, x^2, x^3, sin, cos, sin2, cos2, 1/(1+x^2), exp(-x^2/4).
    # Inspiration 1: P0..P6, sin, cos, sin2.
    # Inspiration 2: T0..T6, sin, cos, sin2.
    # 
    # The current basis is quite diverse.
    # Maybe the issue is scaling or correlation.
    # The current program normalizes by RMS.
    # 
    # Let's try a hybrid: Use Legendre polynomials for the polynomial part (better conditioned on [-1,1] mapped to [-3,3]) and keep the trig/rational/gaussian parts.
    # But we only have 10 slots.
    # Current: 4 poly, 4 trig, 2 special.
    # New: 6 Legendre (P0-P5), 2 Trig (sin, cos), 1 Rational, 1 Gaussian?
    # Or: 5 Legendre (P0-P4), 3 Trig (sin, cos, sin2), 1 Gaussian, 1 Rational?
    # 
    # Let's stick to the "KEEP this design" instruction. The design is "diverse basis normalized by RMS".
    # The "Strategy to try" is a neural network. But the directive says "Do NOT switch to a different design family."
    # So I should NOT implement a neural network. I should stick to the basis expansion.
    # 
    # So, how to improve the basis?
    # 1. Add higher order terms?
    # 2. Improve the special functions?
    # 3. Improve the normalization?
    # 
    # Let's try adding a second Gaussian or a different rational function.
    # Or, let's try the Legendre polynomials for the first 6 slots, as they are orthogonal, which might help BFGS converge better than raw powers.
    # 
    # Let's try:
    # 0: P0 (1)
    # 1: P1 (x/3)
    # 2: P2 ((3(x/3)^2 - 1)/2)
    # 3: P3
    # 4: P4
    # 5: P5
    # 6: sin(x)
    # 7: cos(x)
    # 8: 1/(1+x^2)
    # 9: exp(-x^2/4)
    # 
    # This replaces x, x^2, x^3 with P1, P2, P3 (scaled) and adds P4, P5.
    # P4 and P5 capture higher frequency polynomial behavior.
    # 
    # Let's compute the Legendre polynomials on u = x/3.
    # P0(u) = 1
    # P1(u) = u
    # P2(u) = (3u^2 - 1)/2
    # P3(u) = (5u^3 - 3u)/2
    # P4(u) = (35u^4 - 30u^2 + 3)/8
    # P5(u) = (63u^5 - 70u^3 + 15u)/8
    # 
    # We need to normalize these.
    # RMS of P_k on [-1,1] is 1/sqrt(2k+1).
    # But our domain is [-3,3], so u is in [-1,1].
    # The RMS of P_k(u) on u in [-1,1] is 1/sqrt(2k+1).
    # So we should divide by 1/sqrt(2k+1) to get unit RMS?
    # Wait, the integral of P_k^2(u) from -1 to 1 is 2/(2k+1).
    # Mean square = (1/2) * (2/(2k+1)) = 1/(2k+1).
    # RMS = 1/sqrt(2k+1).
    # So to normalize to RMS 1, we multiply by sqrt(2k+1).
    # 
    # Let's implement this hybrid basis.
    
    # Re-initialize cache for new basis
    if _basis_cache["x"] is not None and np.array_equal(_basis_cache["x"], x):
        V = _basis_cache["V"]
    else:
        n = x.size
        u = x / 3.0
        
        # Legendre Polynomials P0 to P5
        P0 = np.ones(n, dtype=np.float64)
        P1 = u
        P2 = (3.0 * u * u - 1.0) / 2.0
        P3 = (5.0 * u**3 - 3.0 * u) / 2.0
        P4 = (35.0 * u**4 - 30.0 * u * u + 3.0) / 8.0
        P5 = (63.0 * u**5 - 70.0 * u**3 + 15.0 * u) / 8.0
        
        # Trigonometric
        sin_x = np.sin(x)
        cos_x = np.cos(x)
        
        # Special
        rational = 1.0 / (1.0 + x * x)
        gaussian = np.exp(-x * x / 4.0)
        
        V = np.empty((n, 10), dtype=np.float64)
        V[:, 0] = P0
        V[:, 1] = P1
        V[:, 2] = P2
        V[:, 3] = P3
        V[:, 4] = P4
        V[:, 5] = P5
        V[:, 6] = sin_x
        V[:, 7] = cos_x
        V[:, 8] = rational
        V[:, 9] = gaussian
        
        # Normalization
        # P_k: multiply by sqrt(2k+1)
        # sin(x), cos(x): use previous scales
        # rational, gaussian: use previous scales
        # 
        # Scales for P_k:
        # P0: 1
        # P1: sqrt(3)
        # P2: sqrt(5)
        # P3: sqrt(7)
        # P4: sqrt(9) = 3
        # P5: sqrt(11)
        
        scales = np.array([
            1.0,
            np.sqrt(3.0),
            np.sqrt(5.0),
            np.sqrt(7.0),
            3.0,
            np.sqrt(11.0),
            np.sqrt((6.0 - np.sin(6.0)) / 12.0), # sin(x)
            np.sqrt((6.0 + np.sin(6.0)) / 12.0), # cos(x)
            0.508, # rational
            0.645  # gaussian
        ])
        
        V = V / scales
        
        _basis_cache["x"] = x
        _basis_cache["V"] = V
    
    result = V @ p
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
