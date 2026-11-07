import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid RBF-Polynomial Model.
    
    Model:
    f(x) = w1 * exp(-alpha1 * (x - c1)^2) + w2 * exp(-alpha2 * (x - c2)^2) 
           + p0 + p1*x + p2*x^2 + p3*x^3
    
    Parameters (length 10):
    0: w1 (weight of RBF 1)
    1: c1 (center of RBF 1)
    2: a1 (log-inverse-variance of RBF 1, alpha1 = exp(a1))
    3: w2 (weight of RBF 2)
    4: c2 (center of RBF 2)
    5: a2 (log-inverse-variance of RBF 2, alpha2 = exp(a2))
    6: p0 (poly constant)
    7: p1 (poly linear)
    8: p2 (poly quadratic)
    9: p3 (poly cubic)
    
    Using log-inverse-variance ensures alpha is positive and avoids division by zero
    or negative variance issues, providing a smooth landscape for BFGS.
    """
    x = np.asarray(x, dtype=float)
    params = np.asarray(params, dtype=float)
    
    # Ensure we have exactly 10 params, pad or truncate if necessary
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    elif len(params) > 10:
        params = params[:10]
        
    # Unpack parameters
    w1, c1, a1, w2, c2, a2, p0, p1, p2, p3 = params
    
    # Calculate RBF terms
    # alpha = exp(a) ensures alpha > 0
    # We use exp(-alpha * (x-c)^2)
    
    # Safety clamp for alpha to prevent overflow in exp if a is very large
    # exp(20) is ~ 4.8e8, which is fine. exp(100) is huge.
    # If a > 20, alpha is very large, RBF becomes very narrow (delta-like).
    # If a < -20, alpha is very small, RBF becomes very wide (flat).
    
    alpha1 = np.exp(np.clip(a1, -20, 20))
    alpha2 = np.exp(np.clip(a2, -20, 20))
    
    rbf1 = w1 * np.exp(-alpha1 * (x - c1) ** 2)
    rbf2 = w2 * np.exp(-alpha2 * (x - c2) ** 2)
    
    # Calculate Polynomial terms
    poly = p0 + p1 * x + p2 * x**2 + p3 * x**3
    
    result = rbf1 + rbf2 + poly
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
