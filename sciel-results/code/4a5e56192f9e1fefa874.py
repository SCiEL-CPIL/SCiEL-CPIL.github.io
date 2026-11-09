import numpy as np
import time

def equation(b, s, temp, pH, params):
    """
    Closed-form expression for E. coli growth rate.
    Uses a multiplicative biological model with smooth additive correction
    to capture non-multiplicative interactions while maintaining a landscape
    that BFGS can navigate effectively.
    """
    p = params
    
    # Unpack params with smooth positive constraints via softplus-like transforms
    # Use tanh-based scaling for centers to keep them in [0,1]
    T_c = 0.5 * (1.0 + np.tanh(p[0]))
    T_w = 0.1 + 0.4 * (0.5 * (1.0 + np.tanh(p[1])))  # Width in [0.1, 0.5]
    pH_c = 0.5 * (1.0 + np.tanh(p[2]))
    pH_w = 0.1 + 0.4 * (0.5 * (1.0 + np.tanh(p[3])))  # Width in [0.1, 0.5]
    K_s = np.exp(p[4])   # Positive half-saturation constant
    mu_max = np.exp(p[5]) # Positive max rate
    d1 = p[6]
    d2 = p[7]
    d3 = p[8]
    offset = p[9]
    
    # Normalize inputs to [0, 1] based on known ranges
    xb = (b - 1.0) / 4.0
    xT = (temp - 1.0) / 39.0
    xPH = (pH - 2.0) / 10.0
    
    # Monod substrate
    sfac = s / (K_s + s + 1e-9)
    
    # Temperature bell (Gaussian on normalized scale)
    tfac = np.exp(-0.5 * ((xT - T_c) / T_w)**2)
    
    # pH bell (Gaussian on normalized scale)
    pfac = np.exp(-0.5 * ((xPH - pH_c) / pH_w)**2)
    
    # Density modulation (Cubic)
    dfac = 1.0 + d1 * xb + d2 * xb**2 + d3 * xb**3
    
    # Base multiplicative model
    base = mu_max * sfac * tfac * pfac * dfac
    
    # Add smooth correction term that captures interactions
    # This term is smooth and helps BFGS by providing a gradient pathway
    # for deviations from pure multiplicative structure
    # Use normalized features for the correction to keep magnitudes stable
    correction = (
        0.1 * (xT * xPH) +  # T-pH interaction
        0.1 * (xT * xb) +    # T-density interaction
        0.1 * (xPH * xb) +   # pH-density interaction
        0.05 * (xT * xPH * xb)  # 3-way interaction
    )
    
    return base + correction + offset
