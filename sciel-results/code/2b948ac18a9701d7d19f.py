import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Power-law hardening model: sigma = E(T)*eps + K(T)*|eps|^n*sign(eps)
    with linear temperature dependence on E and K.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain)
    temp = np.asarray(temp)
    
    # Elastic modulus with temperature dependence
    E = p0 + p1 * temp
    
    # Hardening coefficient with temperature dependence
    K = p2 + p3 * temp
    
    # Power-law exponent (global constant, constrained to reasonable range)
    n = 1.0 + 3.0 * (1.0 / (1.0 + np.exp(-p4)))  # sigmoid to keep n in (1, 4)
    
    # Core power-law model
    abs_strain = np.abs(strain)
    sign_strain = np.sign(strain)
    
    # Handle zero strain carefully
    safe_strain = np.where(abs_strain > 1e-12, abs_strain, 1e-12)
    power_term = K * np.power(safe_strain, n) * sign_strain
    
    # Elastic term
    elastic_term = E * strain
    
    # Core model
    core = elastic_term + power_term
    
    # Additional correction terms for fine-tuning
    # These provide flexibility for non-ideal behavior
    correction = (
        p5 * strain**2 + 
        p6 * temp + 
        p7 * strain * temp + 
        p8 * temp**2 + 
        p9 * strain * temp**2
    )
    
    return core + correction
