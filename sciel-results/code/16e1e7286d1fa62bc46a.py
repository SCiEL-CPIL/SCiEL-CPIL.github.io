"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using Monod kinetics with logistic temp/pH effects
    and a quadratic correction term.
    
    Structure:
    - Monod: s / (Ks + s) for substrate saturation
    - Logistic temp effect: 1 / (1 + exp(-a*(temp - t_opt)))
    - Logistic pH effect: 1 / (1 + exp(-a2*(ph - ph_opt)))
    - Quadratic correction: c_b*b^2 + c_s*s^2 (captures non-monotonic density effects)
    - Linear temp correction: c_t*(temp - t_opt) (fine-tunes temperature response)
    
    Params (length 10):
    p[0]: Max growth rate (mu_max)
    p[1]: Monod constant Ks
    p[2]: Temperature optimum t_opt
    p[3]: Temperature sensitivity a
    p[4]: pH optimum ph_opt
    p[5]: pH sensitivity a2
    p[6]: Quadratic coeff for b
    p[7]: Quadratic coeff for s
    p[8]: Linear temp correction
    p[9]: Offset
    """
    p = params
    
    mu_max = p[0]
    Ks = np.abs(p[1]) + 1e-8
    t_opt = p[2]
    a_t = p[3]
    ph_opt = p[4]
    a_ph = p[5]
    c_b = p[6]
    c_s = p[7]
    c_t = p[8]
    offset = p[9]
    
    # Monod substrate saturation
    monod = s / (Ks + s)
    
    # Logistic temperature effect (smooth peak around t_opt)
    # Using sigmoid: 1/(1+exp(-a*(temp-t_opt)))
    # To get a peak, we use a symmetric form: exp(-a*(temp-t_opt)^2)
    # But that's Gaussian. Let's use a product of sigmoids for low and high temp cutoffs.
    # Simpler: Gaussian temp effect
    g_t = np.exp(-a_t * (temp - t_opt)**2)
    
    # Gaussian pH effect
    g_ph = np.exp(-a_ph * (pH - ph_opt)**2)
    
    # Main term
    main = mu_max * monod * g_t * g_ph
    
    # Quadratic corrections for density and substrate (non-monotonic effects)
    quad = c_b * b**2 + c_s * s**2
    
    # Linear temperature correction (allows asymmetric response)
    linear_t = c_t * (temp - t_opt)
    
    return main + quad + linear_t + offset
# EVOLVE-BLOCK-END
