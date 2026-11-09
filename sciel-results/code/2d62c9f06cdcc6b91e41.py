import numpy as np
import time
from scipy.optimize import minimize

def equation(b, s, temp, pH, params):
    """
    E. coli growth rate: linear combination of biological basis functions.
    Linear in params -> convex loss, BFGS finds global minimum.
    
    10 basis functions:
    0: bias
    1: ln(1+s)            - Monod-like substrate saturation
    2: s^2/(1+s^2)        - Haldane-like substrate inhibition
    3: (temp-37)          - linear temp
    4: (temp-37)^2        - temp curvature (parabolic optimum)
    5: (pH-7)             - linear pH
    6: (pH-7)^2           - pH curvature
    7: ln(1+b)            - density-dependent inhibition
    8: ln(1+s)*(temp-37)  - substrate-temp interaction
    9: ln(1+s)*(pH-7)     - substrate-pH interaction
    """
    p = params
    
    ln1s = np.log1p(s)
    haldane = (s ** 2) / (1.0 + s ** 2)
    dT = temp - 37.0
    dPh = pH - 7.0
    ln1b = np.log1p(b)
    
    return (p[0]
            + p[1] * ln1s
            + p[2] * haldane
            + p[3] * dT
            + p[4] * dT ** 2
            + p[5] * dPh
            + p[6] * dPh ** 2
            + p[7] * ln1b
            + p[8] * ln1s * dT
            + p[9] * ln1s * dPh)

def _objective(params, b, s, temp, pH, y):
    """MSE objective for fitting."""
    pred = equation(b, s, temp, pH, params)
    if not np.all(np.isfinite(pred)):
        return 1e30
    return np.mean((pred - y) ** 2)

def fit_params(b, s, temp, pH, y, deadline_margin=5.0):
    """Fit parameters with BFGS (convex problem, converges reliably)."""
    start_time = time.time()
    
    # BFGS as required by contract - convex so it finds global min
    x0 = np.ones(10)
    res = minimize(_objective, x0, args=(b, s, temp, pH, y), method='BFGS',
                   options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-12})
    
    elapsed = time.time() - start_time
    if elapsed > 110 - deadline_margin:
        return res.x
    
    # Polish with L-BFGS-B for extra precision
    bounds = [(-10, 10)] * 10
    res2 = minimize(_objective, res.x, args=(b, s, temp, pH, y),
                    method='L-BFGS-B', bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
    
    elapsed = time.time() - start_time
    if elapsed > 110 - deadline_margin:
        return res.x
    
    if res2.fun < res.fun:
        return res2.x
    return res.x
