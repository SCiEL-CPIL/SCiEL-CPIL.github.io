import numpy as np

def equation(b, s, temp, pH, params):
    p = params
    
    # Normalize inputs to [0, 1]
    # b: [1, 5] -> [0, 1]
    xb = (b - 1.0) / 4.0
    # s: [1, 5] -> [0, 1]
    xs = (s - 1.0) / 4.0
    # temp: [1, 40] -> [0, 1]
    xt = (temp - 1.0) / 39.0
    # pH: [2, 12] -> [0, 1]
    xp = (pH - 2.0) / 10.0
    
    # Extract parameters
    mu_max = np.log1p(np.exp(p[0]))  # softplus, ensures positive
    K_s = np.log1p(np.exp(p[1]))     # softplus, ensures positive
    T_opt = p[2]                     # raw, can be any real
    T_w = np.log1p(np.exp(p[3]))     # softplus, ensures positive width
    pH_opt = p[4]                    # raw
    pH_w = np.log1p(np.exp(p[5]))    # softplus, ensures positive width
    d1 = p[6]
    d2 = p[7]
    d3 = p[8]
    offset = p[9]
    
    # Monod
    # Avoid division by zero
    sfac = xs / (K_s + xs + 1e-9)
    
    # Temperature Gaussian
    # Clip T_w to avoid division by zero if softplus returns very small?
    # softplus(1) ~ 1.3, so it's fine.
    tfac = np.exp(-0.5 * ((xt - T_opt) / (T_w + 1e-9))**2)
    
    # pH Gaussian
    pfac = np.exp(-0.5 * ((xp - pH_opt) / (pH_w + 1e-9))**2)
    
    # Density Polynomial
    dfac = 1.0 + d1 * xb + d2 * xb**2 + d3 * xb**3
    
    return mu_max * sfac * tfac * pfac * dfac + offset
