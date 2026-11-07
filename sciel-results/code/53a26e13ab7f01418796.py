"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _build_rational_design_matrix(x, y):
    """
    Build a design matrix for a rational function model.
    
    We use a rational function with quadratic numerator and quadratic denominator:
    (p0 + p1*x + p2*y + p3*x^2 + p4*y^2 + p5*x*y) / (1 + q1*x + q2*y + q3*x^2 + q4*y^2)
    
    To make this work with the linear-in-params framework (BFGS fits params),
    we need to think about this differently. Since BFGS minimizes over params,
    and our equation returns predictions, we can use a different approach.
    
    Actually, the rational function is NOT linear in its parameters when we consider
    both numerator and denominator params. So we need a different strategy.
    
    Instead, let's use a more flexible basis that can capture non-polynomial behavior.
    We'll use a combination of:
    - Polynomial terms up to degree 3
    - Terms like 1/(1+x^2), 1/(1+y^2) to capture saturation/asymptotic behavior
    - Interaction terms
    
    Let's design 10 basis functions that are linear in params:
    0: 1
    1: x
    2: y
    3: x^2
    4: y^2
    5: x*y
    6: 1/(1+x^2)
    7: 1/(1+y^2)
    8: x/(1+x^2)
    9: y/(1+y^2)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    F0 = np.ones_like(x)
    F1 = x
    F2 = y
    F3 = x**2
    F4 = y**2
    F5 = x * y
    F6 = 1.0 / (1.0 + x**2)
    F7 = 1.0 / (1.0 + y**2)
    F8 = x / (1.0 + x**2)
    F9 = y / (1.0 + y**2)
    
    return np.column_stack([F0, F1, F2, F3, F4, F5, F6, F7, F8, F9])

def _build_hybrid_basis(x, y):
    """
    Build a hybrid basis combining polynomial, rational, and trigonometric terms
    to capture a wider range of functional behaviors.
    
    Basis terms (10 total):
    0: 1
    1: x
    2: y
    3: x^2
    4: y^2
    5: x*y
    6: 1/(1+x^2+y^2)  - captures radial decay/saturation
    7: x/(1+x^2+y^2)   - captures directional saturation
    8: y/(1+x^2+y^2)   - captures directional saturation
    9: cos(x*y)         - captures oscillatory interaction
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    r2 = x**2 + y**2
    denom = 1.0 + r2
    
    F0 = np.ones_like(x)
    F1 = x
    F2 = y
    F3 = x**2
    F4 = y**2
    F5 = x * y
    F6 = 1.0 / denom
    F7 = x / denom
    F8 = y / denom
    F9 = np.cos(x * y)
    
    return np.column_stack([F0, F1, F2, F3, F4, F5, F6, F7, F8, F9])

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict y using a linear combination of hybrid basis features.
    The params are fitted by the evaluator using BFGS.
    
    We use a hybrid basis that combines polynomial, rational, and trigonometric
    terms to capture a wide range of functional behaviors including:
    - Polynomial trends
    - Saturation/asymptotic behavior (via rational terms)
    - Oscillatory patterns (via trigonometric terms)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Ensure params has length 10
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    else:
        params = params[:10]
    
    # Build design matrix using hybrid basis
    X = _build_hybrid_basis(x, y)
    
    # Compute prediction: X @ params
    prediction = X @ params
    
    # Ensure prediction is finite
    if not np.all(np.isfinite(prediction)):
        return np.zeros_like(x)
    
    return prediction
# EVOLVE-BLOCK-END
