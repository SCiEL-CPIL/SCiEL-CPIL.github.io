"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT_MARGIN = 148.0  # seconds, keep a safety margin before 150s kill

def _time_left():
    return _start_time + _TIMEOUT_MARGIN - time.time()

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression with 10 parameters that tries to recover an unknown function.
    Strategy: Use a multiplicative/nonlinear product-of-sums basis where parameters
    act as exponents, amplitudes, and frequencies. This allows the model to represent
    terms like x^p * y^q or sin(f*x) directly, rather than just linear combinations
    of pre-computed basis functions. This increases the effective hypothesis space
    for symbolic regression targets that are products or powers.
    
    Basis design (10 params):
    p0: Amplitude 1
    p1: Exponent for x in term 1
    p2: Exponent for y in term 1
    p3: Amplitude 2
    p4: Frequency for sin(x)
    p5: Frequency for sin(y)
    p6: Amplitude 3 (for x*y term)
    p7: Amplitude 4 (for x^2+y^2 term)
    p8: Amplitude 5 (for cos(x*y))
    p9: Offset
    
    Terms:
    1. p0 * x^p1 * y^p2
    2. p3 * sin(p4*x) * sin(p5*y)
    3. p6 * x * y
    4. p7 * (x^2 + y^2)
    5. p8 * cos(x*y)
    6. p9 (constant)
    
    This structure allows BFGS to find that p1=1, p2=1, p0=1 recovers x*y,
    or p1=2, p2=0, p0=1 recovers x^2, etc.
    """
    # Ensure inputs are arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Handle numerical stability for power functions
    # x and y are in [-1, 1]. x^p is problematic if x is negative and p is non-integer.
    # To ensure differentiability and real-valuedness, we use sign-preserving power:
    # sign(x) * |x|^p. For p=1, this is x. For p=2, this is x^2.
    # This is a common trick in symbolic regression to handle negative bases.
    
    # Term 1: p0 * sign(x)^p1 * sign(y)^p2 (approximated as sign(x)*|x|^p1 * sign(y)*|y|^p2)
    # Actually, simpler: use np.sign(x) * np.abs(x)**p1
    # But if p1 is not integer, this is differentiable except at 0. BFGS should handle it.
    # Alternative: Use cos/sin for flexibility, or just polynomial terms with fixed powers
    # but learned coefficients? No, that's the old strategy.
    
    # Let's refine the basis to be more robust.
    # If we use x**p1, and x is negative, it's complex. 
    # Use: sign(x) * abs(x)**p1
    # However, if p1 is close to 0.5, the derivative at 0 is infinite.
    # A safer nonlinear basis that is smooth everywhere:
    # 1. p0 * x * y
    # 2. p1 * x**2
    # 3. p2 * y**2
    # 4. p3 * x**3
    # 5. p4 * y**3
    # 6. p5 * sin(x)
    # 7. p6 * sin(y)
    # 7. p7 * cos(x)
    # 8. p8 * cos(y)
    # 9. p9 * 1
    
    # Wait, the "different strategy" should be materially different.
    # The previous strategy was a linear basis. 
    # A "Product of Sums" or "Neural Network-like" single layer with nonlinear activation 
    # on the inputs is different.
    
    # Let's try a Radial Basis Function (RBF) like approach with 10 params:
    # 5 centers (cx_i, cy_i) and 5 weights (w_i).
    # pred = sum_{i=1 to 5} w_i * exp(-r_i^2 / (2 * sigma^2))
    # But sigma is fixed? Or one sigma?
    # With 10 params: 5 centers (10 params) is too many.
    # 2 centers (4 params) + 2 widths (2 params) + 4 weights (4 params) = 10 params.
    # This is a Gaussian Mixture Model for regression. It is universal approximator.
    
    # Let's implement a 2-component Gaussian Mixture Model.
    # Params:
    # p0, p1: center 1 (cx1, cy1)
    # p2, p3: center 2 (cx2, cy2)
    # p4: width 1 (sigma1)
    # p5: width 2 (sigma2)
    # p6: weight 1 (w1)
    # p7: weight 2 (w2)
    # p8: offset
    # p9: global scale? Or maybe a 3rd small component?
    # Let's stick to 2 components + offset + scale.
    
    cx1, cy1, cx2, cy2, sigma1, sigma2, w1, w2, offset, scale = params
    
    # Ensure sigma is positive to avoid division by zero or negative variance
    # Use softplus or abs + small epsilon
    s1 = np.abs(sigma1) + 1e-6
    s2 = np.abs(sigma2) + 1e-6
    
    # Compute squared distances
    # d1^2 = (x - cx1)^2 + (y - cy1)^2
    dx1 = x - cx1
    dy1 = y - cy1
    d1_sq = dx1 * dx1 + dy1 * dy1
    
    dx2 = x - cx2
    dy2 = y - cy2
    d2_sq = dx2 * dx2 + dy2 * dy2
    
    # Gaussian kernels
    # exp(-d^2 / (2 * sigma^2))
    # To prevent underflow/overflow issues with BFGS, we can clip the exponent
    # but exp of negative number is safe.
    g1 = np.exp(-d1_sq / (2.0 * s1 * s1))
    g2 = np.exp(-d2_sq / (2.0 * s2 * s2))
    
    # Prediction
    pred = scale * (w1 * g1 + w2 * g2 + offset)
    
    # Check for non-finite values
    if not np.all(np.isfinite(pred)):
        # Return a safe large value or zeros to penalize, but task says non-finite costs 1e30
        # We should try to return finite values if possible.
        # If params are huge, exp might underflow to 0, which is fine.
        # If params are NaN, we return NaN.
        # Let's just return pred. The evaluator handles the penalty.
        pass
        
    return pred

# EVOLVE-BLOCK-END
