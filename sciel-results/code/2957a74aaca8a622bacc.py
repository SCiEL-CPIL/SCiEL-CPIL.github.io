import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a scaled polynomial expansion to improve numerical conditioning 
    for the BFGS optimizer. The inputs are normalized to have zero mean 
    and unit variance (approximately) before expansion, which helps the 
    optimizer converge to a better solution compared to raw polynomials.
    
    params (10 constants):
    p0: Intercept
    p1: Linear strain coeff (scaled)
    p2: Quadratic strain coeff (scaled)
    p3: Cubic strain coeff (scaled)
    p4: Linear temp coeff (scaled)
    p5: Strain*Temp interaction (linear strain)
    p6: Strain^2*Temp interaction
    p7: Strain^3*Temp interaction
    p8: Quartic strain coeff (scaled)
    p9: Strain^4*Temp interaction
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    e = strain
    t = temp
    
    # Normalize inputs to improve conditioning
    # Strain range approx [-0.002, 0.96], center ~0.48, width ~0.98
    # Temp range approx [0.06, 1.0], center ~0.53, width ~0.94
    e_norm = (e - 0.48) / 0.98
    t_norm = (t - 0.53) / 0.94
    
    # Pre-compute powers of normalized strain
    e2 = e_norm * e_norm
    e3 = e2 * e_norm
    e4 = e2 * e2
    
    # Basis functions
    # We use a full quadratic in temp? No, only 10 params.
    # Let's use:
    # 1, e, e^2, e^3, e^4
    # t, e*t, e^2*t, e^3*t, e^4*t
    # This covers all combinations up to degree 4 in e and degree 1 in t.
    
    phi0 = np.ones_like(e)
    phi1 = e_norm
    phi2 = e2
    phi3 = e3
    phi4 = e4
    phi5 = t_norm
    phi6 = e_norm * t_norm
    phi7 = e2 * t_norm
    phi8 = e3 * t_norm
    phi9 = e4 * t_norm
    
    stress = (p0 * phi0 + p1 * phi1 + p2 * phi2 + p3 * phi3 + 
              p4 * phi4 + p5 * phi5 + p6 * phi6 + p7 * phi7 + 
              p8 * phi8 + p9 * phi9)
              
    return stress
