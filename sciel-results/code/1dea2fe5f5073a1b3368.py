import numpy as np
import time
from scipy.optimize import minimize

def _softplus(x):
    """Numerically stable softplus: log(1+exp(x)) without overflow."""
    return np.where(x > 20, x, np.log1p(np.exp(x)))

def equation(b, s, temp, pH, params):
    p = params
    
    # Extract parameters
    c0      = p[0]
    c1      = p[1]
    c2      = p[2]
    c3      = p[3]
    c4      = p[4]
    mu_max  = _softplus(p[5])
    K_s     = _softplus(p[6]) + 1e-6
    T_opt   = p[7]
    pH_opt  = p[8]
    T_width = _softplus(p[9]) + 0.1
    
    # Normalizations
    # b: [1, 5] -> [0, 1]
    bn = (b - 1.0) / 4.0
    # s: [1, 40] -> [0, 1] for Monod input? No, Monod uses raw s usually, but K_s will adapt.
    # Let's normalize s for the linear part if we had one, but here s is only in Monod.
    # temp: [1, 40] -> [0, 1] approx. (1-20)/19? Let's use (temp-20)/10.
    tn = (temp - 20.0) / 10.0
    # pH: [2, 12] -> (pH-7)/2.
    pn = (pH - 7.0) / 2.0
    
    # Linear/Interaction Background
    # This part is linear in params, easy to fit.
    # It captures general trends.
    bg = c0 + c1 * bn + c2 * tn + c3 * pn + c4 * bn * tn
    
    # Biological Peak
    # Monod for substrate
    sfac = s / (K_s + s)
    
    # Gaussian for Temperature
    # T_width is in normalized units (tn).
    tfac = np.exp(-0.5 * ((tn - T_opt) / T_width)**2)
    
    # Gaussian for pH
    # Fixed width 1.0 in normalized units (pn).
    pfac = np.exp(-0.5 * ((pn - pH_opt) / 1.0)**2)
    
    # Combine
    # The bio term is additive to the background.
    # This structure allows the background to offset the peak if needed,
    # or handle cases where growth is not zero even at low T/pH (if mu_max is small).
    return bg + mu_max * sfac * tfac * pfac

def _objective(params, b, s, temp, pH, y):
    """MSE objective for fitting."""
    pred = equation(b, s, temp, pH, params)
    if not np.all(np.isfinite(pred)):
        return 1e30
    mse = np.mean((pred - y) ** 2)
    # Small regularization
    reg = 1e-6 * np.sum(params**2)
    return mse + reg

def fit_params(b, s, temp, pH, y, deadline_margin=5.0):
    """Fit parameters with initial BFGS + final L-BFGS-B polish pass."""
    start_time = time.time()
    
    # Initial fit with BFGS as required by contract
    x0 = np.ones(10)
    res = minimize(_objective, x0, args=(b, s, temp, pH, y), method='BFGS')
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 110 - deadline_margin:
        return res.x
    
    # Final high-precision polish with L-BFGS-B (bounded, more robust)
    # Use tight bounds to keep parameters in reasonable ranges
    bounds = [
        (-10, 10),   # c0
        (-10, 10),   # c1
        (-10, 10),   # c2
        (-10, 10),   # c3
        (-10, 10),   # c4
        (-5, 10),    # mu_max (softplus input)
        (-5, 10),    # K_s (softplus input)
        (-2, 4),     # T_opt (normalized)
        (-2, 2),     # pH_opt (normalized)
        (-2, 5),     # T_width (softplus input)
    ]
    
    res2 = minimize(_objective, res.x, args=(b, s, temp, pH, y), 
                    method='L-BFGS-B', bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
    
    # Check deadline again
    elapsed = time.time() - start_time
    if elapsed > 110 - deadline_margin:
        return res.x  # Return safer initial result if close to deadline
    
    # Use the better result
    if res2.fun < res.fun:
        return res2.x
    return res.x
