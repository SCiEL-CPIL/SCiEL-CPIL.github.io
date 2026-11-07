import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for an aluminium rod using a Voce-type exponential hardening
    formulation combined with a smooth Ramberg-Osgood-like correction.
    
    sigma = E(T)*u + sigma_y(T) + Q(T)*(1 - exp(-u/H(T)))
    
    Where:
      E(T) = p0 + p1*T          (Elastic modulus)
      sigma_y(T) = p2 + p3*T    (Yield stress)
      Q(T) = p4 + p5*T          (Saturation stress range)
      H(T) = softplus(p6) + softplus(p7)*T  (Hardening modulus, constrained positive)
      p8, p9: Additional polynomial correction terms for flexibility
              correction = (p8 + p9*T) * u^2 * (1 + 0.1*T)
    
    This structure naturally captures the asymptotic saturation of stress at high strains,
    a key feature of aluminium that pure power-law models fail to represent. The added
    polynomial term provides additional flexibility for fitting nuances in the data.
    
    Parameters (10):
    p0, p1: Elastic modulus coefficients
    p2, p3: Yield stress coefficients
    p4, p5: Saturation stress coefficients
    p6, p7: Hardening modulus coefficients (via softplus to ensure positivity)
    p8, p9: Polynomial correction coefficients
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    u = strain
    T = temp
    
    # Helper for stable softplus to avoid overflow/underflow and ensure differentiability
    def stable_softplus(x):
        # For x > 20, softplus(x) ~ x
        # For x <= 20, softplus(x) = log(1 + exp(x))
        return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -50, 50))))
    
    # Temperature-dependent coefficients
    # E(T): Elastic modulus
    E = p0 + p1 * T
    
    # sigma_y(T): Yield stress
    sigma_y = p2 + p3 * T
    
    # Q(T): Saturation stress range
    Q = p4 + p5 * T
    
    # H(T): Hardening modulus (constrained positive via softplus)
    H_const = stable_softplus(p6)
    H_temp_coeff = stable_softplus(p7)
    H = H_const + H_temp_coeff * T
    
    # Ensure H is not too small to avoid numerical issues
    H_safe = np.maximum(H, 1e-6)
    
    # Voce hardening term: Q * (1 - exp(-u/H))
    # For numerical stability, clip the argument
    arg = -u / H_safe
    arg_clipped = np.clip(arg, -50, 50)
    voce_term = Q * (1.0 - np.exp(arg_clipped))
    
    # Elastic contribution
    elastic = E * u
    
    # Polynomial correction for additional flexibility
    # This helps capture any residual non-Voce behavior
    u2 = u * u
    correction = (p8 + p9 * T) * u2 * (1.0 + 0.1 * T)
    
    # Combine all terms
    sigma = elastic + sigma_y + voce_term + correction
    
    return sigma
