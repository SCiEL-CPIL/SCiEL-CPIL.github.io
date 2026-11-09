import numpy as np
import time
from scipy.optimize import minimize

def equation(b, s, temp, pH, params):
    """
    E. coli growth rate model: linear combination of biological basis functions.
    Linear in params -> convex loss for BFGS.
    
    Basis functions (10 total):
    0: bias
    1: ln(1+s)           - substrate saturation (Monod-like)
    2: temp              - linear temp
    3: (temp-37)^2       - temp curvature (optimum ~37C)
    4: pH                - linear pH
    5: (pH-7)^2          - pH curvature (optimum ~7)
    6: ln(1+b)           - density saturation
    7: ln(1+s)*(temp-37) - temp-substrate interaction
    8: ln(1+s)*(pH-7)    - pH-substrate interaction
    9: ln(1+b)*(temp-37) - density-temp interaction
    """
    p = params
    
    # Precompute basis functions
    log_s = np.log1p(s)
    t_dev = temp - 37.0
    ph_dev = pH - 7.0
    log_b = np.log1p(b)
    
    B0 = np.ones_like(s)
    B1 = log_s
    B2 = temp
    B3 = t_dev * t_dev
    B4 = pH
    B5 = ph_dev * ph_dev
    B6 = log_b
    B7 = log_s * t_dev
    B8 = log_s * ph_dev
    B9 = log_b * t_dev
    
    return (p[0] * B0 + p[1] * B1 + p[2] * B2 + p[3] * B3 +
            p[4] * B4 + p[5] * B5 + p[6] * B6 + p[7] * B7 +
            p[8] * B8 + p[9] * B9)

def _objective(params, b, s, temp, pH, y):
    """MSE objective for fitting."""
    pred = equation(b, s, temp, pH, params)
    if not np.all(np.isfinite(pred)):
        return 1e30
    return np.mean((pred - y) ** 2)

def fit_params(b, s, temp, pH, y, deadline_margin=5.0):
    """Fit parameters with BFGS."""
    start_time = time.time()
    
    x0 = np.ones(10)
    res = minimize(_objective, x0, args=(b, s, temp, pH, y), method='BFGS',
                   options={'maxiter': 1000, 'gtol': 1e-8})
    
    elapsed = time.time() - start_time
    if elapsed > 110 - deadline_margin:
        return res.x
    
    return res.x
