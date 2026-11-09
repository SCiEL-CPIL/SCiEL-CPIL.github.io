import numpy as np

def equation(b, s, temp, pH, params):
    p = params
    
    # Unpack params with scaled initialization to start in a "sensible" region
    # p[0]: T_center (tanh scaled to 0.5 +/- 0.4)
    T_c = 0.5 + 0.4 * np.tanh(p[0])
    
    # p[1]: T_width (exp scaled, starts ~0.54)
    T_w = 0.2 * np.exp(p[1])
    
    # p[2]: pH_center (tanh scaled to 0.5 +/- 0.4)
    pH_c = 0.5 + 0.4 * np.tanh(p[2])
    
    # p[3]: pH_width (exp scaled, starts ~0.54)
    pH_w = 0.2 * np.exp(p[3])
    
    # p[4]: K_s (exp scaled, starts ~2.7)
    K_s = 1.0 * np.exp(p[4])
    
    # p[5]: mu_max (exp scaled, starts ~2.7)
    mu_max = 1.0 * np.exp(p[5])
    
    # p[6]: d1 (linear coeff for b)
    d1 = p[6]
    # p[7]: d2 (quad coeff for b)
    d2 = p[7]
    # p[8]: d3 (cubic coeff for b)
    d3 = p[8]
    # p[9]: offset
    offset = p[9]
    
    # Normalize inputs to [0, 1] based on known ranges
    # b: [1, 5]
    xb = (b - 1.0) / 4.0
    # s: [1, 5]
    # T: [1, 40]
    xT = (temp - 1.0) / 39.0
    # pH: [2, 12]
    xPH = (pH - 2.0) / 10.0
    
    # Calculate components
    # Monod substrate
    sfac = s / (K_s + s + 1e-9)
    
    # Temperature bell (Gaussian on normalized scale)
    tfac = np.exp(-0.5 * ((xT - T_c) / T_w)**2)
    
    # pH bell (Gaussian on normalized scale)
    pfac = np.exp(-0.5 * ((xPH - pH_c) / pH_w)**2)
    
    # Density modulation (Cubic)
    dfac = 1.0 + d1 * xb + d2 * xb**2 + d3 * xb**3
    
    # Final prediction
    return mu_max * sfac * tfac * pfac * dfac + offset
