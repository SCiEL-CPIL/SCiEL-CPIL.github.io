import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear-in-parameters basis expansion for stress-strain of an aluminium rod.
    
    sigma = sum(p_i * phi_i(strain, temp))
    
    Basis functions:
    phi_0 = 1
    phi_1 = strain
    phi_2 = strain * temp
    phi_3 = strain * temp^2
    phi_4 = strain * (strain^2 + delta)^(0.5)  # ~ strain^3 for large strain, smooth at 0
    phi_5 = strain * (strain^2 + delta)^(1.0)  # ~ strain^4
    phi_6 = strain * (strain^2 + delta)^(1.5)  # ~ strain^5
    phi_7 = strain * (strain^2 + delta)^(0.5) * temp
    phi_8 = strain * (strain^2 + delta)^(1.0) * temp
    phi_9 = temp^2
    
    This formulation is linear in the parameters, making the optimization landscape convex (or nearly so),
    which allows BFGS to find the global minimum reliably and quickly.
    """
    p = params
    
    # Ensure inputs are float arrays
    strain = np.asarray(strain, dtype=float)
    temp = np.asarray(temp, dtype=float)
    
    # If parameters are not finite, return zeros to avoid propagation of NaNs/Infs
    if not np.all(np.isfinite(p)):
        return np.zeros_like(strain)
        
    # Small constant for numerical stability at strain=0
    delta = 1e-8
    
    # Precompute smooth non-linear strain terms
    s2 = strain * strain
    base = s2 + delta
    
    # Hardening terms (odd functions, smooth)
    # phi_4: approx strain^3
    h1 = strain * np.power(base, 0.5)
    # phi_5: approx strain^4? No, strain * (strain^2)^1 = strain^3. Wait.
    # strain * (s2)^0.5 = strain * |strain| = strain^2 * sign(strain).
    # Let's check powers:
    # base^0.5 ~ |strain|. strain * |strain| = strain^2 * sign(strain). This is continuous but derivative discontinuity at 0?
    # d/ds (s|s|) = 2|s|. Continuous.
    # base^1.0 ~ s2. strain * s2 = s^3. Smooth.
    # base^1.5 ~ s3. strain * s3 = s^4 * sign(s).
    
    # Let's use exponents that result in smooth integer-like powers for large strain
    # e1 = 0.5 -> s * |s| ~ s^2 (odd)
    # e2 = 1.0 -> s * s^2 ~ s^3 (odd)
    # e3 = 1.5 -> s * s^3 ~ s^4 (odd? No, s * s^3 = s^4 is even? No. s * (s^2)^1.5 = s * |s|^3. If s>0, s^4. If s<0, -s^4. So it's odd.)
    
    h2 = strain * np.power(base, 1.0)
    h3 = strain * np.power(base, 1.5)
    
    # Build the design matrix terms
    # We can compute the prediction directly as a linear combination
    
    # phi_0
    term0 = 1.0
    # phi_1
    term1 = strain
    # phi_2
    term2 = strain * temp
    # phi_3
    term3 = strain * temp * temp
    # phi_4
    term4 = h1
    # phi_5
    term5 = h2
    # phi_6
    term6 = h3
    # phi_7
    term7 = h1 * temp
    # phi_8
    term8 = h2 * temp
    # phi_9
    term9 = temp * temp
    
    # Linear combination
    sigma = (p[0] * term0 + 
             p[1] * term1 + 
             p[2] * term2 + 
             p[3] * term3 + 
             p[4] * term4 + 
             p[5] * term5 + 
             p[6] * term6 + 
             p[7] * term7 + 
             p[8] * term8 + 
             p[9] * term9)
             
    return sigma
